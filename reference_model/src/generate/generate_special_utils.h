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

#ifndef GENERATE_SPECIAL_UTILS_H_
#define GENERATE_SPECIAL_UTILS_H_

#include "dtype_limits.h"
#include "generate_utils.h"
#include "half.hpp"

#include <map>
#include <random>

using namespace TosaReference;

/// \brief Get a value that will overflow (positive) the TosaRefType if DataType allows it.
template <TOSA_REF_TYPE TosaRefType, typename DataType>
DataType aboveMax()
{
    // Using * 2 to get a value "larger than max" as a simple way to guarantee
    // that the result will not get rounded down to TosaRefType::max.
    // If we can't fit that value in DataType, then we just return the maximum
    // finite value.
    if (double(std::numeric_limits<DataType>::max()) >= double(DtypeLimits<TosaRefType>::max) * 2)
        return static_cast<DataType>(DtypeLimits<TosaRefType>::max) * static_cast<DataType>(2);
    else
        return std::numeric_limits<DataType>::max();
}

/// \brief Get a value that will overflow (negative) the TosaRefType if DataType allows it.
template <TOSA_REF_TYPE TosaRefType, typename DataType>
DataType belowLowest()
{
    // Using * 2 to get a value "lower than lowest" as a simple way to guarantee
    // that the result will not get rounded up to TosaRefType::lowest.
    // If we can't fit that value in DataType, then we just return the lowest
    // finite value.
    if (double(std::numeric_limits<DataType>::lowest()) <= double(DtypeLimits<TosaRefType>::lowest) * 2)
        return static_cast<DataType>(DtypeLimits<TosaRefType>::lowest) * static_cast<DataType>(2);
    else
        return std::numeric_limits<DataType>::lowest();
}

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
            WARNING("[Generator][S] Bad random range - min:%g max:%g", static_cast<double>(min),
                    static_cast<double>(max));
            return false;
        }
        return true;
    }

    bool _checkEven(DataType val)
    {
        if ((val != static_cast<DataType>(round(val))) || (val / 2.0 != round(val / 2.0)))
        {
            WARNING("[Generator][S] Bad even integer generated - %g", static_cast<double>(val));
            return false;
        }
        return true;
    }

    bool _checkOdd(DataType val)
    {
        if ((val != static_cast<DataType>(round(val))) || (val / 2.0 == round(val / 2.0)))
        {
            WARNING("[Generator][S] Bad odd integer generated - %g", static_cast<double>(val));
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
                    WARNING("[Generator][S] Unsupported integer type.");
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
                WARNING("[Generator][S] Unsupported integer type.");
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
        Min,       // Smallest positive normal floating point value
        Max,       // Largest positive floating point value
        Lowest,    // Smallest normal value in the dtype
        One,
        Two,
        Ten,
        Euler,         // Floating point number
        Pythagoras,    // Floating point number
        MinDenorm,     // Smallest positive denormal floating point value
        ULPMax,        // To force overflows to infinity when added/subtracted
        RndFloat,
        RndInteger,
        RndEvenInteger,
        RndOddInteger,
        AboveMaxINT8,
        AboveMaxINT16,
        AboveMaxINT32,
        AboveMaxFP8E4M3,
        AboveMaxFP8E5M2,
        AboveMaxBF16,
        AboveMaxFP16,
        AboveMaxFP32,
        BelowLowestINT8,
        BelowLowestINT16,
        BelowLowestINT32,
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
    DataType evaluate(RandomGen<DataType> rng) const
    {
        // Work out the simple values
        switch (_value)
        {
            case Zero:
            case Inf:
            case NaN:
            case Min:
            case Max:
            case Lowest:
            case One:
            case Two:
            case Ten:
            case Euler:
            case Pythagoras:
            case MinDenorm:
            case ULPMax:
            case AboveMaxINT8:
            case AboveMaxINT16:
            case AboveMaxINT32:
            case AboveMaxFP8E4M3:
            case AboveMaxFP8E5M2:
            case AboveMaxBF16:
            case AboveMaxFP16:
            case AboveMaxFP32:
            case BelowLowestINT8:
            case BelowLowestINT16:
            case BelowLowestINT32:
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
                WARNING("[Generator][S] Unsupported special value type.");
                return static_cast<DataType>(0.0);
        }
        return _negative ? -rnd : rnd;
    }

private:
    template <typename DataType>
    DataType _static_evaluate(SpecialValsEnum v, bool negate) const
    {
        // Work out the static value
        switch (v)
        {
            case Zero:
                return negate ? -static_cast<DataType>(0) : static_cast<DataType>(0);
            case Inf:
                return negate ? -std::numeric_limits<DataType>::infinity() : std::numeric_limits<DataType>::infinity();
            case NaN:
                return std::numeric_limits<DataType>::quiet_NaN();
            case Min:
                return negate ? -std::numeric_limits<DataType>::min() : std::numeric_limits<DataType>::min();
            case Max:
                return negate ? -std::numeric_limits<DataType>::max() : std::numeric_limits<DataType>::max();
            case Lowest:
                return negate ? -std::numeric_limits<DataType>::lowest() : std::numeric_limits<DataType>::lowest();
            case One:
                return static_cast<DataType>(negate ? -1 : 1);
            case Two:
                return static_cast<DataType>(negate ? -2 : 2);
            case Ten:
                return static_cast<DataType>(negate ? -10 : 10);
            case Euler:
                return static_cast<DataType>(negate ? -2.71828 : 2.71828);
            case Pythagoras:
                return static_cast<DataType>(negate ? -1.41421 : 1.41421);
            case MinDenorm:
                if (!std::is_same<DataType, fp8e4m3>::value && !std::is_same<DataType, fp8e5m2>::value)
                {
                    // TODO: Re-enable subnorm testing
                    // Do not test subnormal values for anything but FP8
                    // as they are allowed to be flushed to zero and are
                    // not currently supported by Conformance Testing
                    return negate ? -static_cast<DataType>(0) : static_cast<DataType>(0);
                }
                else
                {
                    return negate ? -std::numeric_limits<DataType>::denorm_min()
                                  : std::numeric_limits<DataType>::denorm_min();
                }
            case ULPMax: {
                DataType max = std::numeric_limits<DataType>::max();
                DataType ulp = max - nextafter(max, static_cast<DataType>(0));
                return negate ? -ulp : ulp;
            }
            case AboveMaxINT8: {
                DataType above_max = aboveMax<TOSA_REF_TYPE_INT8, DataType>();
                return negate ? -above_max : above_max;
            }
            case AboveMaxINT16: {
                DataType above_max = aboveMax<TOSA_REF_TYPE_INT16, DataType>();
                return negate ? -above_max : above_max;
            }

            case AboveMaxINT32: {
                DataType above_max = aboveMax<TOSA_REF_TYPE_INT32, DataType>();
                return negate ? -above_max : above_max;
            }

            case AboveMaxFP8E4M3: {
                DataType above_max = aboveMax<TOSA_REF_TYPE_FP8E4M3, DataType>();
                return negate ? -above_max : above_max;
            }
            case AboveMaxFP8E5M2: {
                DataType above_max = aboveMax<TOSA_REF_TYPE_FP8E5M2, DataType>();
                return negate ? -above_max : above_max;
            }
            case AboveMaxBF16: {
                DataType above_max = aboveMax<TOSA_REF_TYPE_BF16, DataType>();
                return negate ? -above_max : above_max;
            }
            case AboveMaxFP16: {
                DataType above_max = aboveMax<TOSA_REF_TYPE_FP16, DataType>();
                return negate ? -above_max : above_max;
            }
            case AboveMaxFP32: {
                DataType above_max = aboveMax<TOSA_REF_TYPE_FP32, DataType>();
                return negate ? -above_max : above_max;
            }
            case BelowLowestINT8: {
                DataType below_lowest = belowLowest<TOSA_REF_TYPE_INT8, DataType>();
                return negate ? -below_lowest : below_lowest;
            }
            case BelowLowestINT16: {
                DataType below_lowest = belowLowest<TOSA_REF_TYPE_INT16, DataType>();
                return negate ? -below_lowest : below_lowest;
            }
            case BelowLowestINT32: {
                DataType below_lowest = belowLowest<TOSA_REF_TYPE_INT32, DataType>();
                return negate ? -below_lowest : below_lowest;
            }
            default:
                // Assumption that we only get called with a valid enum
                return static_cast<DataType>(0);
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
T: Number of test cases defined for the op.
   Only the values that fit in the input tensor will be used. The minimum
   guaranteed size for input tensors is defined by the TOSA_SPECIAL_MIN_SIZE
   value in the TosaTestGen python class.

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

#endif    // GENERATE_SPECIAL_UTILS_H_

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

#ifndef GENERATE_SPECIAL_UTILS_H_
#define GENERATE_SPECIAL_UTILS_H_

#include "cfloat.h"
#include "dtype_limits.h"
#include "generate_utils.h"
#include "half.hpp"

#include <map>
#include <random>

using namespace TosaReference;

// TODO(ITL): numeric_limits is not constexpr for cfloat.h types, so the compiler cannot check that
// we are not in fact ever overflowing here and will raise a warning. Silence the warnings for now,
// and try to make numeric_limits constexpr in the future, although it's a bit hard without
// bit_cast which is C++20.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wimplicit-const-int-float-conversion"
#pragma clang diagnostic ignored "-Winteger-overflow"
#pragma clang diagnostic ignored "-Wconstant-conversion"
#endif

/// \brief Get a value that will overflow (positive) the TosaRefType if DataType allows it.
template <TOSA_REF_TYPE TosaRefType, typename DataType>
DataType aboveMax()
{
    // Using * 2 to get a value "larger than max" as a simple way to guarantee
    // that the result will not get rounded down to TosaRefType::max.
    // If we can't fit that value in DataType, then we just return the maximum
    // finite value.
    if (double(std::numeric_limits<DataType>::max()) >= double(DtypeLimits<TosaRefType>::max) * 2)
        return ct::compat::cast<DataType>(DtypeLimits<TosaRefType>::max) * static_cast<DataType>(2);
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

// Recover the previous settings for diagnostics
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

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
        if constexpr (std::is_integral<DataType>())
        {
            auto dis     = std::uniform_int_distribution<int64_t>(static_cast<int64_t>(min), static_cast<int64_t>(max));
            DataType rnd = static_cast<DataType>(dis(_gen));
            return rnd;
        }
        else
        {
            return _getFloatInteger(min, max, Any);
        }
    }

    DataType getEvenInteger(DataType min, DataType max)
    {
        ASSERT_MSG(!std::is_integral<DataType>(), "Untested usage of getEvenInteger using integral types")
        return _getFloatInteger(min, max, Even);
    }

    DataType getOddInteger(DataType min, DataType max)
    {
        ASSERT_MSG(!std::is_integral<DataType>(), "Untested usage of getEvenInteger using integral types")
        return _getFloatInteger(min, max, Odd);
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

    DataType _getFloatInteger(DataType min, DataType max, _integerTypeEnum type)
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
        MaxShift,          // Number of bits in datatype minus 1
        RndSignInteger,    // From negative number to positive number range
        SixtyTwo,
        Half,
        ThreeHalves,
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

    template <TOSA_REF_TYPE TosaRefType, typename DataType>
    DataType evaluate(RandomGen<DataType>& rng) const
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
            case MaxShift:
            case SixtyTwo:
            case Half:
            case ThreeHalves:
                return _static_evaluate<TosaRefType, DataType>(_value, _negative);
            default:
                // Handle the Random and unsupported cases below
                break;
        }
        // Must be random value, work out positive range
        auto min = _static_evaluate<TosaRefType, DataType>(_rangeMin, false);
        auto max = _static_evaluate<TosaRefType, DataType>(_rangeMax, false);

        DataType rnd;
        switch (_value)
        {
            case RndFloat:
                rnd = rng.getFloat(min, max);
                break;
            case RndInteger:
                rnd = rng.getInteger(min, max);
                break;
            case RndSignInteger:
                // Negative min to positive max
                rnd = rng.getInteger(-min, max);
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
    template <TOSA_REF_TYPE TosaRefType, typename DataType>
    DataType _static_evaluate(SpecialValsEnum v, bool negate) const
    {
        // Work out the static value
        DataType rawVal;
        switch (v)
        {
            case Zero:
                rawVal = static_cast<DataType>(0);
                break;
            case Inf:
                rawVal = DtypeLimits<TosaRefType>::infinity;
                break;
            case NaN:
                rawVal = DtypeLimits<TosaRefType>::quiet_NaN;
                break;
            case Min:
                rawVal = DtypeLimits<TosaRefType>::min;
                break;
            case Max:
                rawVal = DtypeLimits<TosaRefType>::max;
                break;
            case Lowest:
                rawVal = DtypeLimits<TosaRefType>::lowest;
                break;
            case Half:
                rawVal = static_cast<DataType>(0.5);
                break;
            case One:
                rawVal = static_cast<DataType>(1);
                break;
            case ThreeHalves:
                rawVal = static_cast<DataType>(1.5);
                break;
            case Two:
                rawVal = static_cast<DataType>(2);
                break;
            case Ten:
                rawVal = static_cast<DataType>(10);
                break;
            case SixtyTwo:
                rawVal = static_cast<DataType>(62);
                break;
            case Euler:
                rawVal = static_cast<DataType>(2.71828);
                break;
            case Pythagoras:
                rawVal = static_cast<DataType>(1.41421);
                break;
            case MinDenorm:
                if constexpr (!std::is_same<DataType, fp8e4m3>::value && !std::is_same<DataType, fp8e5m2>::value)
                {
                    // TODO: Re-enable subnorm testing
                    // Do not test subnormal values for anything but FP8
                    // as they are allowed to be flushed to zero and are
                    // not currently supported by Conformance Testing
                    rawVal = static_cast<DataType>(0);
                }
                else
                {
                    rawVal = DtypeLimits<TosaRefType>::denorm_min;
                }
                break;
            case ULPMax: {
                DataType max = DtypeLimits<TosaRefType>::max;
                DataType ulp = max - nextafter(max, static_cast<DataType>(0));
                rawVal       = ulp;
                break;
            }
            case AboveMaxINT8: {
                DataType above_max = aboveMax<TOSA_REF_TYPE_INT8, DataType>();
                rawVal             = above_max;
                break;
            }
            case AboveMaxINT16: {
                DataType above_max = aboveMax<TOSA_REF_TYPE_INT16, DataType>();
                rawVal             = above_max;
                break;
            }

            case AboveMaxINT32: {
                DataType above_max = aboveMax<TOSA_REF_TYPE_INT32, DataType>();
                rawVal             = above_max;
                break;
            }

            case AboveMaxFP8E4M3: {
                DataType above_max = aboveMax<TOSA_REF_TYPE_FP8E4M3, DataType>();
                rawVal             = above_max;
                break;
            }
            case AboveMaxFP8E5M2: {
                DataType above_max = aboveMax<TOSA_REF_TYPE_FP8E5M2, DataType>();
                rawVal             = above_max;
                break;
            }
            case AboveMaxBF16: {
                DataType above_max = aboveMax<TOSA_REF_TYPE_BF16, DataType>();
                rawVal             = above_max;
                break;
            }
            case AboveMaxFP16: {
                DataType above_max = aboveMax<TOSA_REF_TYPE_FP16, DataType>();
                rawVal             = above_max;
                break;
            }
            case AboveMaxFP32: {
                DataType above_max = aboveMax<TOSA_REF_TYPE_FP32, DataType>();
                rawVal             = above_max;
                break;
            }
            case BelowLowestINT8: {
                DataType below_lowest = belowLowest<TOSA_REF_TYPE_INT8, DataType>();
                rawVal                = below_lowest;
                break;
            }
            case BelowLowestINT16: {
                DataType below_lowest = belowLowest<TOSA_REF_TYPE_INT16, DataType>();
                rawVal                = below_lowest;
                break;
            }
            case BelowLowestINT32: {
                DataType below_lowest = belowLowest<TOSA_REF_TYPE_INT32, DataType>();
                rawVal                = below_lowest;
                break;
            }
            case MaxShift: {
                return static_cast<DataType>(sizeof(DataType) * 8 - 1);
            }
            default:
                // Assumption that we only get called with a valid enum
                rawVal = static_cast<DataType>(0);
        }

        if constexpr (std::is_same_v<DataType, uint8_t> || std::is_same_v<DataType, uint16_t>)
        {
            // No negative values allowed in unsigned, return 0 instead
            return negate ? static_cast<DataType>(0) : rawVal;
        }
        else if (!(std::isinf(rawVal)) && (-double(rawVal) > DtypeLimits<TosaRefType>::max ||
                                           -double(rawVal) < DtypeLimits<TosaRefType>::lowest))
        {
            // Make sure an integer value does not overflow if negated
            return rawVal;
        }
        else
        {
            return negate ? -rawVal : rawVal;
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

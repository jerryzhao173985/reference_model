
// Copyright (c) 2020, ARM Limited.
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

#ifndef TOSA_REFERENCE_QUANT_UTIL_H
#define TOSA_REFERENCE_QUANT_UTIL_H

#include "arith_util.h"
#include "func_debug.h"
#include "ops/template_types.h"
#include "tosa_generated.h"

using namespace tosa;

namespace TosaReference
{

class QuantUtil
{
public:
    static void reciprocal_scale(int32_t value,
                                 // Output
                                 int32_t& multiplier,
                                 int32_t& shift)
    {
        ASSERT_MSG(value > 0,
                   "AvgPool2d reciprocal_scale() error: # of elements should be > 1 but is %d", value);
        uint32_t value_u32 = (uint32_t)value;
        int32_t k          = 32 - LEADING_ZEROS_32(value_u32 - 1);    // (1<<k)/2 < value <= (1<<k)
        int64_t numerator  = ((1L << 30) + 1) << k;
        multiplier         = numerator / value;    // (1<<30) <= multiplier < (1<<31)
        shift              = 30 + k;
    }

    static int32_t apply_scale_32(int32_t value, int32_t multiplier, int32_t shift, bool double_round = true)
    {
        ASSERT_MSG(multiplier >= 0, "apply_scale_32() error: multiplier should >= 0 but is %d", multiplier);
        ASSERT_MSG(shift >= 2 && shift <= 62, "apply_scale_32() error: shift should be within [2, 62] but is %d",
                   shift);
        int64_t round = 1L << (shift - 1);
        if (double_round)
        {
            if (shift > 31 && value >= 0)
                round += (1L << 30);
            if (shift > 31 && value < 0)
                round -= (1L << 30);
        }
        int64_t result = (int64_t)value * multiplier + round;
        result         = result >> shift;
        ASSERT_MSG(result >= -(1L << 31) && result < (1L << 31),
                   "apply_scale_32() error: scaled result exceed int32 numeric range");
        return static_cast<int32_t>(result);
    }
};

class TypeChecker
{
public:
    static bool is_integer(DType dtype)
    {
        if (dtype == DType_INT4 || dtype == DType_INT8 || dtype == DType_AINT8 || dtype == DType_UINT8 ||
            dtype == DType_INT16 || dtype == DType_INT32 || dtype == DType_INT48)
        {
            return true;
        }
        return false;
    }
    static bool is_symmetric(DType dtype)
    {
        if (dtype == DType_INT4 || dtype == DType_INT8 || dtype == DType_INT16 || dtype == DType_INT32 ||
            dtype == DType_INT48)
        {
            return true;
        }
        return false;
    }
};
};    // namespace TosaReference

#endif

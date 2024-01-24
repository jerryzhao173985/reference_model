
// Copyright (c) 2020-2023, ARM Limited.
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

/*
 *   Filename:     src/arith_util.h
 *   Description:
 *    arithmetic utility macro, include:
 *      fp16 (float16_t ) type alias
 *      bitwise operation
 *      fix point arithmetic
 *      fp16 type conversion(in binary translation)
 *      fp16 arithmetic (disguised with fp32 now)
 */

#ifndef ARITH_UTIL_H
#define ARITH_UTIL_H

#include <fenv.h>
#include <math.h>
#define __STDC_LIMIT_MACROS    //enable min/max of plain data type
#include "dtype.h"
#include "func_config.h"
#include "func_debug.h"
#include "half.hpp"
#include "inttypes.h"
#include <Eigen/Core>
#include <bitset>
#include <cassert>
#include <iostream>
#include <limits>
#include <stdint.h>
#include <typeinfo>

using namespace tosa;
using namespace std;
using namespace TosaReference;

inline size_t _count_one(uint64_t val)
{
    size_t count = 0;
    for (; val; count++)
    {
        val &= val - 1;
    }
    return count;
}

template <typename T>
inline size_t _integer_log2(T val)
{
    size_t result = 0;
    while (val >>= 1)
    {
        ++result;
    }
    return result;
}

template <typename T>
inline size_t _count_leading_zeros(T val)
{
    size_t size  = sizeof(T) * 8;
    size_t count = 0;
    T msb        = static_cast<T>(1) << (size - 1);
    for (size_t i = 0; i < size; i++)
    {
        if (!((val << i) & msb))
            count++;
        else
            break;
    }
    return count;
}

template <typename T>
inline size_t _count_leading_ones(T val)
{
    size_t size  = sizeof(T) * 8;
    size_t count = 0;
    T msb        = static_cast<T>(1) << (size - 1);
    for (size_t i = 0; i < size; i++)
    {
        if ((val << i) & msb)
            count++;
        else
            break;
    }
    return count;
}

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
// Compute ceiling of (a/b)
#define DIV_CEIL(a, b) ((a) % (b) ? ((a) / (b) + 1) : ((a) / (b)))

// Returns a mask of 1's of this size
#define ONES_MASK(SIZE) ((uint64_t)((SIZE) >= 64 ? UINT64_C(0xffffffffffffffff) : (UINT64_C(1) << (SIZE)) - 1))

// Returns a field of bits from HIGH_BIT to LOW_BIT, right-shifted
// include both side, equivalent VAL[LOW_BIT:HIGH_BIT] in verilog

#define BIT_FIELD(HIGH_BIT, LOW_BIT, VAL) (((uint64_t)(VAL) >> (LOW_BIT)) & ONES_MASK((HIGH_BIT) + 1 - (LOW_BIT)))

// Returns a bit at a particular position
#define BIT_EXTRACT(POS, VAL) (((uint64_t)(VAL) >> (POS)) & (1))

// Use Brian Kernigahan's way: https://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetKernighan
// Does this need to support floating point type?
// Not sure if static_cast is the right thing to do, try to be type safe first
#define ONES_COUNT(VAL) (_count_one((uint64_t)(VAL)))

#define SHIFT(SHF, VAL) (((SHF) > 0) ? ((VAL) << (SHF)) : ((SHF < 0) ? ((VAL) >> (-(SHF))) : (VAL)))
#define ROUNDTO(A, B) ((A) % (B) == 0 ? (A) : ((A) / (B) + 1) * (B))
#define ROUNDTOLOWER(A, B) (((A) / (B)) * (B))
#define BIDIRECTIONAL_SHIFT(VAL, SHIFT) (((SHIFT) >= 0) ? ((VAL) << (SHIFT)) : ((VAL) >> (-(SHIFT))))
#define ILOG2(VAL) (_integer_log2(VAL))

// Get negative value (2's complement)
#define NEGATIVE_8(VAL) ((uint8_t)(~(VAL) + 1))
#define NEGATIVE_16(VAL) ((uint16_t)(~(VAL) + 1))
#define NEGATIVE_32(VAL) ((uint32_t)(~(VAL) + 1))
#define NEGATIVE_64(VAL) ((uint64_t)(~(VAL) + 1))
// Convert a bit quanity to the minimum bytes required to hold those bits
#define BITS_TO_BYTES(BITS) (ROUNDTO((BITS), 8) / 8)

// Count leading zeros/ones for 8/16/32/64-bit operands
// (I don't see an obvious way to collapse this into a size-independent set)
// treated as unsigned
#define LEADING_ZEROS_64(VAL) (_count_leading_zeros((uint64_t)(VAL)))
#define LEADING_ZEROS_32(VAL) (_count_leading_zeros((uint32_t)(VAL)))
#define LEADING_ZEROS_16(VAL) (_count_leading_zeros((uint16_t)(VAL)))
#define LEADING_ZEROS_8(VAL) (_count_leading_zeros((uint8_t)(VAL)))
#define LEADING_ZEROS(VAL) (_count_leading_zeros(VAL))

#define LEADING_ONES_64(VAL) _count_leading_ones((uint64_t)(VAL))
#define LEADING_ONES_32(VAL) _count_leading_ones((uint32_t)(VAL))
#define LEADING_ONES_16(VAL) _count_leading_ones((uint16_t)(VAL))
#define LEADING_ONES_8(VAL) _count_leading_ones((uint8_t)(VAL))
#define LEADING_ONES(VAL) _count_leading_ones(VAL)
// math operation
// sign-extended for signed version
// extend different return type (8, 16, 32) + (S, U)
// Saturate a value at a certain bitwidth, signed and unsigned versions
// Format is as followed: SATURATE_VAL_{saturation_sign}_{return_type}
// for example
// SATURATE_VAL_U_8U(8,300) will return uint8_t with value of 255(0xff)
// SATURATE_VAL_S_32S(5,-48) will return int32_t with value of -16(0x10)
// note that negative value can cast to unsigned return type using native uint(int) cast
// so SATURATE_VAL_S_8U(5,-40) will have value 0'b1110000 which is in turn 224 in uint8_t

template <typename T>
constexpr T bitmask(const uint32_t width)
{
    ASSERT(width <= sizeof(T) * 8);
    return width == sizeof(T) * 8 ? static_cast<T>(std::numeric_limits<uintmax_t>::max())
                                  : (static_cast<T>(1) << width) - 1;
}

template <typename T>
constexpr T minval(const uint32_t width)
{
    ASSERT(width <= sizeof(T) * 8);
    return std::is_signed<T>::value ? -(static_cast<T>(1) << (width - 1)) : 0;
}

template <typename T>
constexpr T maxval(const uint32_t width)
{
    ASSERT(width <= sizeof(T) * 8);
    return bitmask<T>(width - std::is_signed<T>::value);
}

template <typename T>
constexpr T saturate(const uint32_t width, const intmax_t value)
{
    // clang-format off
    return  static_cast<T>(
        std::min(
            std::max(
                value,
                static_cast<intmax_t>(minval<T>(width))
            ),
            static_cast<intmax_t>(maxval<T>(width))
        )
    );
    // clang-format on
}

inline void float_trunc_bytes(float* src)
{
    /* Set the least significant two bytes to zero for the input float value.*/
    uint32_t* ptr = reinterpret_cast<uint32_t*>(src);
    *ptr          = *ptr & UINT32_C(0xffff0000);
}

inline void truncateFloatToBFloat(float* src, int64_t size)
{
    /* Set the least significant two bytes to zero for each float
    value in the input src buffer. */
    ASSERT_MEM(src);
    ASSERT_MSG(size > 0, "Size of src (representing number of values in src) must be a positive integer.");
    for (; size != 0; src++, size--)
    {
        float_trunc_bytes(src);
    }
}

inline bool checkValidBFloat(float src)
{
    /* Checks if the least significant two bytes are zero. */
    uint32_t* ptr = reinterpret_cast<uint32_t*>(&src);
    return (*ptr & UINT32_C(0x0000ffff)) == 0;
}

template <TOSA_REF_TYPE Dtype>
float fpTrunc(float f_in)
{
    /* Truncates a float value based on the TOSA_REF_TYPE it represents.*/
    switch (Dtype)
    {
        case TOSA_REF_TYPE_BF16:
            truncateFloatToBFloat(&f_in, 1);
            break;
        case TOSA_REF_TYPE_FP16:
            // Cast to temporary float16 value before casting back to float32
            {
                half_float::half h = half_float::half_cast<half_float::half, float>(f_in);
                f_in               = half_float::half_cast<float, half_float::half>(h);
                break;
            }
        case TOSA_REF_TYPE_FP32:
            // No-op for fp32
            break;
        default:
            ASSERT_MSG(false, "TOSA_REF_TYPE %s should not be float-truncated.", EnumNameTOSAREFTYPE(Dtype));
    }
    return f_in;
}

#endif /* _ARITH_UTIL_H */

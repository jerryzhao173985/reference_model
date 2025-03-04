
// Copyright (c) 2020-2025, ARM Limited.
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
 *    and include the arithmetic helpers listed in Section 4.3.1. of the spec
 */

#ifndef ARITH_UTIL_H
#define ARITH_UTIL_H

#include "dtype.h"
#include "dtype_limits.h"
#include "func_config.h"
#include "func_debug.h"
#include "half.hpp"
#include "inttypes.h"
#include "ops/template_types.h"
#include "subgraph_traverser.h"
#include <bitset>
#include <cassert>
#include <fenv.h>
#include <limits>
#include <stdint.h>
#include <typeinfo>

using namespace tosa;
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
        // Rounding first in order to prevent unrepresentative BF16 number
        *src = static_cast<float>(static_cast<bf16>(*src));
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
            ASSERT_MSG(false, "TOSA_REF_TYPE %s should not be float-cast.", EnumNameTOSAREFTYPE(Dtype));
    }
    return f_in;
}

// truncate an integer to the width of DType
template <typename T, TOSA_REF_TYPE OutDType>
T intTrunc(T in)
{
    constexpr int width = GetNumBits<OutDType>().value;

    // If GetNumBits is not implemented and returns 0, the code is unsafe.
    if constexpr (width > 0)
    {
        static_assert(std::is_integral_v<T>, "intTrunc can only be called with integer inputs");
        static_assert(width < sizeof(T) * 8, "intTrunc output should be narrower than the input");

        constexpr T mask = bitmask<T>(width);

        in &= mask;

        // sign-extend back to the original size
        const T sign = in & (1 << (width - 1));
        if (sign)
            in |= (~mask);
    }
    else
    {
        FATAL_ERROR("Internal error in the reference model unsupported output type for intTrunc");
    }

    // Only reachable if we did not cause a FATAL_ERROR
    return in;
}

// return the maximum value when interpreting type T as a signed value.
template <TOSA_REF_TYPE Dtype>
int32_t getSignedMaximum()
{
    if (Dtype == TOSA_REF_TYPE_INT8 || Dtype == TOSA_REF_TYPE_UINT8)
        return GetQMax<TOSA_REF_TYPE_INT8>::value;

    if (Dtype == TOSA_REF_TYPE_INT16 || Dtype == TOSA_REF_TYPE_UINT16)
        return GetQMax<TOSA_REF_TYPE_INT16>::value;

    if (Dtype == TOSA_REF_TYPE_INT32)
        return GetQMax<TOSA_REF_TYPE_INT32>::value;

    FATAL_ERROR("Get maximum_s for the dtype input is not supported");
    return 0;
}

// return the minimum value when interpreting type T as a signed value.
template <TOSA_REF_TYPE Dtype>
int32_t getSignedMinimum()
{
    if (Dtype == TOSA_REF_TYPE_INT8 || Dtype == TOSA_REF_TYPE_UINT8)
        return GetQMin<TOSA_REF_TYPE_INT8>::value;

    if (Dtype == TOSA_REF_TYPE_INT16 || Dtype == TOSA_REF_TYPE_UINT16)
        return GetQMin<TOSA_REF_TYPE_INT16>::value;

    if (Dtype == TOSA_REF_TYPE_INT32)
        return GetQMin<TOSA_REF_TYPE_INT32>::value;

    FATAL_ERROR("Get minimum_s for the dtype input is not supported");
    return 0;
}

// return the maximum value when interpreting type T as an unsigned value.
template <TOSA_REF_TYPE Dtype>
int32_t getUnsignedMaximum()
{
    if (Dtype == TOSA_REF_TYPE_INT8 || Dtype == TOSA_REF_TYPE_UINT8)
        return GetQMax<TOSA_REF_TYPE_UINT8>::value;

    if (Dtype == TOSA_REF_TYPE_INT16 || Dtype == TOSA_REF_TYPE_UINT16)
        return GetQMax<TOSA_REF_TYPE_UINT16>::value;

    if (Dtype == TOSA_REF_TYPE_INT32)
        return std::numeric_limits<uint32_t>::max();

    FATAL_ERROR("Get maximum_u for the dtype input is not supported");
    return 0;
}

// return the minimum value when interpreting type T as an unsigned value.
template <TOSA_REF_TYPE Dtype>
int32_t getUnsignedMinimum()
{
    if (Dtype == TOSA_REF_TYPE_INT8 || Dtype == TOSA_REF_TYPE_UINT8)
        return GetQMin<TOSA_REF_TYPE_UINT8>::value;

    if (Dtype == TOSA_REF_TYPE_INT16 || Dtype == TOSA_REF_TYPE_UINT16)
        return GetQMin<TOSA_REF_TYPE_UINT16>::value;

    if (Dtype == TOSA_REF_TYPE_INT32)
        return std::numeric_limits<uint32_t>::min();

    FATAL_ERROR("Get minimum_u for the dtype input is not supported");
    return 0;
}

inline bool isPropagatingNan(NanPropagationMode nan_mode)
{
    return nan_mode == NanPropagationMode_PROPAGATE;
}

inline bool isIgnoringNan(NanPropagationMode nan_mode)
{
    return nan_mode == NanPropagationMode_IGNORE;
}

template <typename T>
T compareNan(T a, T b, NanPropagationMode nan_mode)
{
    constexpr bool is_floating_point = std::is_floating_point<T>::value;
    if (is_floating_point)
    {
        ASSERT_MSG(std::isnan(a) || std::isnan(b), "Call with no NaN operands is illegal");
        ASSERT_MSG(isPropagatingNan(nan_mode) || isIgnoringNan(nan_mode), "Invalid NaN propagation mode");

        if (isPropagatingNan(nan_mode))
        {
            return NAN;
        }

        // Non NaN Propagation
        return std::isnan(a) ? b : a;
    }

    static_assert(is_floating_point, "Call with integer operands is illegal");
    return 0;
}

template <typename T>
T applyMax(T a, T b, NanPropagationMode nan_mode = NanPropagationMode_PROPAGATE)
{
    if constexpr (std::is_floating_point<T>::value)
    {
        if (std::isnan(a) || std::isnan(b))
        {
            return compareNan(a, b, nan_mode);
        }
    }
    return (a >= b) ? a : b;
}

template <typename T>
T applyMin(T a, T b, NanPropagationMode nan_mode = NanPropagationMode_PROPAGATE)
{
    if constexpr (std::is_floating_point<T>::value)
    {
        if (std::isnan(a) || std::isnan(b))
        {
            return compareNan(a, b, nan_mode);
        }
    }
    return (a < b) ? a : b;
}

// Clip the input value of type T into the range [min, max] of type U, and return the result as type T.
template <typename T, typename U>
T applyClip(T value,
            U min_val,
            U max_val,
            TosaReference::SubgraphTraverser* sgt,
            NanPropagationMode nan_mode = NanPropagationMode_PROPAGATE)
{
    assert(sgt != nullptr);
    assert(sizeof(T) == sizeof(U));
    REQUIRE_SIMPLE(sgt, min_val <= max_val, "min_val is greater than max_val");
    if (std::is_floating_point<U>::value)
        REQUIRE_SIMPLE(sgt, !(std::isnan(min_val) || std::isnan(max_val)), "Operand min and max cannot be NaN");

    value = applyMax<T>(value, min_val, nan_mode);

    // Handle the numbers of an unsigned type U that becomes unrepresentable when type casting to signed.
    if (std::is_signed_v<T> && std::is_unsigned_v<U> && max_val > static_cast<U>(std::numeric_limits<T>::max()))
    {
        max_val = std::numeric_limits<T>::max();
    }

    value = applyMin<T>(value, max_val, nan_mode);

    return value;
}

template <typename T>
T arithRshift(T a, T b)
{
    static_assert(sizeof(int32_t) >= sizeof(T));
    int32_t c = static_cast<int32_t>(a) >> static_cast<int32_t>(b);
    return static_cast<T>(c);
}

// Return the value that will function as lowest for applyMax in the given nan_mode.
// This is useful for padding and for initializing reducers.
template <TOSA_REF_TYPE Dtype, typename T>
constexpr T getApplyMaxPadding(NanPropagationMode nan_mode = NanPropagationMode_PROPAGATE)
{
    // When ignoring NaNs, NaN functions as the *smallest* value for applyMax.
    const auto v =
        (IsFloat<Dtype>() && isIgnoringNan(nan_mode)) ? DtypeLimits<Dtype>::quiet_NaN : DtypeLimits<Dtype>::low_extreme;
    return static_cast<T>(v);
}

// Return the value that will function as highest for applyMin in the given nan_mode.
// This is useful for padding and for initializing reducers.
template <TOSA_REF_TYPE Dtype, typename T>
constexpr T getApplyMinPadding(NanPropagationMode nan_mode = NanPropagationMode_PROPAGATE)
{
    // When ignoring NaNs, NaN functions as the *largest* value for applyMin.
    const auto v = (IsFloat<Dtype>() && isIgnoringNan(nan_mode)) ? DtypeLimits<Dtype>::quiet_NaN
                                                                 : DtypeLimits<Dtype>::high_extreme;
    return static_cast<T>(v);
}

#endif /* _ARITH_UTIL_H */

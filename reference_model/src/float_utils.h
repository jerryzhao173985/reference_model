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

#ifndef FLOAT_UTILS_H_
#define FLOAT_UTILS_H_

#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>
#if defined(__cpp_lib_bit_cast)
#include <bit>
#endif    // defined(__cpp_lib_bit_cast)

namespace tosa::reference::internal
{

namespace float_support
{

struct hidden
{};

#if defined(__cpp_lib_bit_cast)
#define BITCAST_CONSTEXPR constexpr inline

constexpr inline int32_t get_bits(const float& f)
{
    return std::bit_cast<int32_t>(f);
}
constexpr inline float from_bits(const int32_t& i)
{
    return std::bit_cast<float>(i);
}

#else
#define BITCAST_CONSTEXPR inline

union ufloat32
{
    constexpr ufloat32(const float& x)
        : f(x)
    {}
    constexpr ufloat32(const int32_t& x)
        : i(x)
    {}

    float f;
    int32_t i;
};

inline int32_t get_bits(const float& f)
{
    return ufloat32(f).i;
}
inline float from_bits(const int32_t& i)
{
    return ufloat32(i).f;
}
#endif

}    // namespace float_support

template <typename storage_t,
          size_t n_exp_bits,
          bool has_nan,
          bool with_denorm,
          bool with_infinity,
          std::enable_if_t<(n_exp_bits + 1 < sizeof(storage_t) * 8), bool> = true>
class float_t
{
    storage_t m_data = 0;

public:
    static constexpr size_t n_exponent_bits    = n_exp_bits;
    static constexpr size_t n_significand_bits = sizeof(storage_t) * 8 - 1 - n_exp_bits;
    static constexpr int64_t exponent_bias     = (1 << (n_exp_bits - 1)) - 1;

    /// \brief Construct a floating point type with the given bit
    /// representation.
    static constexpr float_t from_bits(storage_t bits)
    {
        return float_t(float_support::hidden(), bits);
    }

    /// \brief Construct a float from the given sign, exponent and significand
    /// bits.
    static constexpr float_t from_bits(bool pm, storage_t e, storage_t s)
    {
        storage_t bits = pm ? 1 : 0;

        bits <<= n_exp_bits;
        bits |= e;

        bits <<= n_significand_bits;
        if (with_denorm || e)
            bits |= s;

        return float_t(float_support::hidden(), bits);
    }

    /// \brief (Hidden) Construct a float type from a given bit pattern
    constexpr float_t(const float_support::hidden&, storage_t bits)
        : m_data(bits)
    {}

    constexpr float_t()
        : m_data(0)
    {}
    constexpr float_t(const float_t& other)
        : m_data(other.m_data)
    {}

    /// \brief Cast to a different floating point representation.
    template <typename other_storage_t,
              size_t other_n_exp_bits,
              bool other_has_nan,
              bool other_has_denorm,
              bool other_has_infinity>
    constexpr inline
        operator float_t<other_storage_t, other_n_exp_bits, other_has_nan, other_has_denorm, other_has_infinity>() const
    {
        using other_float_t =
            float_t<other_storage_t, other_n_exp_bits, other_has_nan, other_has_denorm, other_has_infinity>;

        // Shortcut for types which are fundamentally similar (e.g., bf16 ->
        // fp32)
        if constexpr (n_exp_bits == other_n_exp_bits && sizeof(other_storage_t) >= sizeof(storage_t) &&
                      has_nan == other_has_nan)
        {
            return other_float_t::from_bits(static_cast<other_storage_t>(m_data)
                                            << (sizeof(other_storage_t) - sizeof(storage_t)) * 8);
        }

        // Get initial values for the new floating point type
        const bool sign_bit       = m_data < 0;
        int64_t new_exponent_bits = 0;
        uint64_t new_significand  = 0;

        if (is_nan() || is_infinity())
        {
            new_exponent_bits = (1 << other_n_exp_bits) - 1;

            if (is_nan())
            {
                if constexpr (other_has_infinity)
                {
                    // Copy across the `not_quiet bit`; set the LSB. Don't
                    // attempt to copy across any of the rest of the payload.
                    new_significand =
                        0x1 | (((significand() >> (n_significand_bits - 1)) & 1) << other_float_t::n_significand_bits);
                }
                else
                {
                    new_significand = (1ul << other_float_t::n_significand_bits) - 1;
                }
            }
            else if constexpr (!other_has_infinity)
            {
                new_significand = (1ul << other_float_t::n_significand_bits) - (other_has_nan ? 2 : 1);
            }
        }
        else if (!is_zero())
        {
            const int64_t this_exponent_bits = exponent_bits();
            {
                constexpr int64_t exponent_rebias = other_float_t::exponent_bias - exponent_bias;
                new_exponent_bits                 = std::max(this_exponent_bits + exponent_rebias, exponent_rebias + 1);
            }
            new_significand = this->significand() << (64 - n_significand_bits);

            // Normalise subnormals
            if (this_exponent_bits == 0)
            {
                // Shift the most-significant 1 out of the magnitude to convert
                // it to a significand. Modify the exponent accordingly.
                uint8_t shift = __builtin_clzl(new_significand) + 1;
                new_exponent_bits -= shift;
                new_significand <<= shift;
            }

            // Align the significand for the output type
            uint32_t shift                = 64 - other_float_t::n_significand_bits;
            const bool other_is_subnormal = new_exponent_bits <= 0;
            if (other_is_subnormal)
            {
                shift += 1 - new_exponent_bits;
                new_exponent_bits = 0;
            }

            const uint64_t shift_out = shift == 64 ? new_significand : new_significand & ((1ll << shift) - 1);
            new_significand          = shift == 64 ? 0 : new_significand >> shift;

            // Reinsert the most-significant-one if this is a subnormal in the
            // output type.
            new_significand |= (other_is_subnormal ? 1ll : 0) << (64 - shift);

            // Apply rounding based on the bits shifted out of the significand
            const uint64_t shift_half = 1ll << (shift - 1);
            if (shift_out > shift_half || (shift_out == shift_half && (new_significand & 1)))
            {
                new_significand += 1;

                // Handle the case that the significand overflowed due to
                // rounding
                constexpr uint64_t max_significand = (1ll << other_float_t::n_significand_bits) - 1;
                if (new_significand > max_significand)
                {
                    new_significand = 0;
                    new_exponent_bits++;
                }
            }

            // Saturate to infinity if the exponent is larger than can be
            // represented in the output type. This can only occur if the size
            // of the exponent of the new type is not greater than the exponent
            // of the old type.
            if constexpr (other_n_exp_bits <= n_exp_bits)
            {
                constexpr int64_t inf_exp_bits = (1ll << other_n_exp_bits) - 1;
                if (new_exponent_bits >= inf_exp_bits)
                {
                    new_exponent_bits = inf_exp_bits;
                    new_significand =
                        other_has_infinity ? 0 : (1ul << other_float_t::n_significand_bits) - (other_has_nan ? 2 : 1);
                }
            }
        }

        return other_float_t::from_bits(sign_bit, new_exponent_bits, new_significand);
    }

    /// \brief Convert from a 32-bit floating point value
    BITCAST_CONSTEXPR
    float_t(const float& f)
    {
        // If this format exactly represents the binary32 format then get
        // the bits from the provided float; otherwise get a binary32
        // representation and then convert to this format.
        if constexpr (represents_binary32())
            m_data = float_support::get_bits(f);
        else
            m_data = static_cast<float_t<storage_t, n_exp_bits, has_nan, with_denorm, with_infinity>>(
                         static_cast<float_t<int32_t, 8, true, true, true>>(f))
                         .m_data;
    }

    /// \brief Cast to a 32-bit floating point value
    BITCAST_CONSTEXPR operator float() const
    {
        // If this format exactly represents the binary32 format then return
        // a float; otherwise get a binary32 representation and then return
        // a float.
        if constexpr (represents_binary32())
            return float_support::from_bits(m_data);
        else
            return static_cast<float>(this->operator float_t<int32_t, 8, true, true, true>());
    }

    /// \brief Return whether this type represents the IEEE754 binary32
    /// format
    constexpr static inline bool represents_binary32()
    {
        return std::is_same_v<storage_t, int32_t> && n_exp_bits == 8 && has_nan && with_denorm && with_infinity;
    }

    constexpr auto operator-() const
    {
        return from_bits(m_data ^ (1ll << (sizeof(storage_t) * 8 - 1)));
    }

    constexpr bool is_subnormal() const
    {
        return exponent_bits() == 0 && significand() != 0;
    }

    constexpr bool is_zero() const
    {
        return exponent_bits() == 0 && significand() == 0;
    }

    constexpr bool is_nan() const
    {
        return has_nan && (exponent_bits() == (1ul << n_exponent_bits) - 1) &&
               ((with_infinity && significand()) ||
                (!with_infinity && significand() == (1ul << n_significand_bits) - 1));
    }

    constexpr bool is_infinity() const
    {
        return with_infinity && ((exponent_bits() == (1ul << n_exponent_bits) - 1) && !significand());
    }

    constexpr inline const storage_t& bits() const
    {
        return m_data;
    }

    /// \brief Get the exponent
    constexpr inline int64_t exponent() const
    {
        return std::max<int64_t>(exponent_bits(), 1ul) - exponent_bias;
    }

    /// \brief Get the bits from the exponent
    constexpr inline uint64_t exponent_bits() const
    {
        constexpr uint64_t mask = (1ul << n_exp_bits) - 1;
        return (m_data >> n_significand_bits) & mask;
    }

    constexpr inline uint64_t significand() const
    {
        return m_data & ((1ul << n_significand_bits) - 1);
    }

    constexpr inline bool operator==(const float_t& other) const
    {
        return !is_nan() && !other.is_nan() && ((is_zero() && other.is_zero()) || bits() == other.bits());
    }

    constexpr inline float_t& operator+=(const float_t& rhs)
    {
        this->m_data = static_cast<float_t>(static_cast<float>(*this) + static_cast<float>(rhs)).bits();
        return *this;
    }
};

// This should probably be exported so we can use it elsewhere
#undef BITCAST_CONSTEXPR

namespace float_support
{

// Pre-C++23 these can't be computed as constexpr, so have to hardcode them

template <int>
struct digits10;    // floor(log10(2) * (digits - 1)
template <int>
struct max_digits10;    // ceil(log10(2) * digits + 1)
template <int>
struct min_exponent10;    // floor(log10(2) * min_exponent)
template <int>
struct max_exponent10;    // floor(log10(2) * max_exponent)

template <>
struct digits10<8>
{
    constexpr static inline int value = 2;
};

template <>
struct max_digits10<8>
{
    constexpr static inline int value = 4;
};

template <>
struct digits10<10>
{
    constexpr static inline int value = 2;
};

template <>
struct max_digits10<10>
{
    constexpr static inline int value = 5;
};

template <>
struct digits10<24>
{
    constexpr static inline int value = 6;
};

template <>
struct max_digits10<24>
{
    constexpr static inline int value = 9;
};

template <>
struct min_exponent10<-13>
{
    constexpr static inline int value = -3;
};

template <>
struct max_exponent10<16>
{
    constexpr static inline int value = 4;
};

template <>
struct min_exponent10<-125>
{
    constexpr static inline int value = -37;
};

template <>
struct max_exponent10<128>
{
    constexpr static inline int value = 38;
};

template <int d>
inline constexpr int digits10_v = digits10<d>::value;
template <int d>
inline constexpr int max_digits10_v = max_digits10<d>::value;

template <int e>
inline constexpr int min_exponent10_v = min_exponent10<e>::value;

template <int e>
inline constexpr int max_exponent10_v = max_exponent10<e>::value;

}    // namespace float_support

}    // namespace tosa::reference::internal

namespace std
{

template <typename storage_t, size_t n_exp_bits, bool has_nan, bool has_denorm, bool has_inf>
struct is_floating_point<tosa::reference::internal::float_t<storage_t, n_exp_bits, has_nan, has_denorm, has_inf>>
    : std::integral_constant<bool, true>
{};

template <typename storage_t, size_t n_exp_bits, bool has_nan, bool with_denorm, bool with_inf>
class numeric_limits<tosa::reference::internal::float_t<storage_t, n_exp_bits, has_nan, with_denorm, with_inf>>
{
    using this_float_t = tosa::reference::internal::float_t<storage_t, n_exp_bits, has_nan, with_denorm, with_inf>;

public:
    static constexpr bool is_specialized = true;

    static constexpr auto min() noexcept
    {
        return this_float_t::from_bits(false, 1, 0);
    }

    static constexpr auto max() noexcept
    {
        return this_float_t::from_bits(false, (1 << this_float_t::n_exponent_bits) - 2,
                                       (1 << this_float_t::n_significand_bits) - 1);
    }

    static constexpr auto lowest() noexcept
    {
        return -max();
    }

    static constexpr int digits       = this_float_t::n_significand_bits + 1;
    static constexpr int digits10     = tosa::reference::internal::float_support::digits10_v<digits>;
    static constexpr int max_digits10 = tosa::reference::internal::float_support::max_digits10_v<digits>;

    static constexpr bool is_signed  = true;
    static constexpr bool is_integer = false;
    static constexpr bool is_exact   = false;
    static constexpr int radix       = 2;

    static constexpr auto epsilon() noexcept
    {
        return this_float_t::from_bits(false, this_float_t::exponent_bias - this_float_t::n_significand_bits, 0);
    }

    static constexpr auto round_error() noexcept
    {
        return this_float_t::from_bits(0, this_float_t::exponent_bias - 1, 0);
    }

    static constexpr int min_exponent   = (1 - this_float_t::exponent_bias) + 1;
    static constexpr int min_exponent10 = tosa::reference::internal::float_support::min_exponent10_v<min_exponent>;
    static constexpr int max_exponent   = this_float_t::exponent_bias + 1;
    static constexpr int max_exponent10 = tosa::reference::internal::float_support::max_exponent10_v<max_exponent>;

    static constexpr bool has_infinity             = with_inf;
    static constexpr bool has_quiet_NaN            = has_nan;
    static constexpr bool has_signaling_NaN        = true;
    static constexpr float_denorm_style has_denorm = with_denorm ? denorm_present : denorm_absent;
    static constexpr bool has_denorm_loss          = false;

    static constexpr auto infinity() noexcept
    {
        if constexpr (with_inf)
        {
            return this_float_t::from_bits(false, (1 << this_float_t::n_exponent_bits) - 1, 0);
        }
        else
        {
            return this_float_t::from_bits(false, 0, 0);
        }
    }

    static constexpr auto quiet_NaN() noexcept
    {
        return this_float_t::from_bits(false, (1 << this_float_t::n_exponent_bits) - 1,
                                       1 << (this_float_t::n_significand_bits - 1) | 1);
    }

    static constexpr auto signaling_NaN() noexcept
    {
        return this_float_t::from_bits(false, (1 << this_float_t::n_exponent_bits) - 1, 1);
    }

    static constexpr auto denorm_min() noexcept
    {
        return this_float_t::from_bits(false, 0, 1);
    }

    static constexpr bool is_iec559  = false;
    static constexpr bool is_bounded = false;
    static constexpr bool is_modulo  = false;

    static constexpr bool traps                    = false;
    static constexpr bool tinyness_before          = false;
    static constexpr float_round_style round_style = round_to_nearest;
};

}    // namespace std

#endif    //  _FLOAT_UTILS_H_

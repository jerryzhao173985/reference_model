#include "cfloat.h"
#include "dtype.h"
#include "half.hpp"

using bf16    = ct::cfloat<int16_t, 8, true, true, true>;
using fp8e4m3 = ct::cfloat<int8_t, 4, true, true, false>;
using fp8e5m2 = ct::cfloat<int8_t, 5, true, true, true>;

using half = half_float::half;

namespace TosaReference
{

template <TOSA_REF_TYPE type>
struct DtypeLimits;

template <>
struct DtypeLimits<TOSA_REF_TYPE_FP64>
{
    static constexpr double lowest       = std::numeric_limits<double>::lowest();
    static constexpr double max          = std::numeric_limits<double>::max();
    static constexpr double min          = std::numeric_limits<double>::min();
    static constexpr double denorm_min   = std::numeric_limits<double>::denorm_min();
    static constexpr double infinity     = std::numeric_limits<double>::infinity();
    static constexpr double low_extreme  = -infinity;
    static constexpr double high_extreme = infinity;
    static constexpr bool has_infinity   = true;
};

template <>
struct DtypeLimits<TOSA_REF_TYPE_FP32>
{
    static constexpr float lowest       = std::numeric_limits<float>::lowest();
    static constexpr float max          = std::numeric_limits<float>::max();
    static constexpr float min          = std::numeric_limits<float>::min();
    static constexpr float denorm_min   = std::numeric_limits<float>::denorm_min();
    static constexpr float infinity     = std::numeric_limits<float>::infinity();
    static constexpr float low_extreme  = -infinity;
    static constexpr float high_extreme = infinity;
    static constexpr bool has_infinity  = true;
};

template <>
struct DtypeLimits<TOSA_REF_TYPE_FP16>
{
    static constexpr half lowest       = std::numeric_limits<half>::lowest();
    static constexpr half max          = std::numeric_limits<half>::max();
    static constexpr half min          = std::numeric_limits<half>::min();
    static constexpr half denorm_min   = std::numeric_limits<half>::denorm_min();
    static constexpr half infinity     = std::numeric_limits<half>::infinity();
    static constexpr half low_extreme  = -infinity;
    static constexpr half high_extreme = infinity;
    static constexpr bool has_infinity = true;
};

template <>
struct DtypeLimits<TOSA_REF_TYPE_BF16>
{
    static constexpr bf16 lowest       = std::numeric_limits<bf16>::lowest();
    static constexpr bf16 max          = std::numeric_limits<bf16>::max();
    static constexpr bf16 min          = std::numeric_limits<bf16>::min();
    static constexpr bf16 denorm_min   = std::numeric_limits<bf16>::denorm_min();
    static constexpr bf16 infinity     = std::numeric_limits<bf16>::infinity();
    static constexpr bf16 low_extreme  = -infinity;
    static constexpr bf16 high_extreme = infinity;
    static constexpr bool has_infinity = true;
};

template <>
struct DtypeLimits<TOSA_REF_TYPE_FP8E4M3>
{
    static constexpr fp8e4m3 lowest       = std::numeric_limits<fp8e4m3>::lowest();
    static constexpr fp8e4m3 max          = std::numeric_limits<fp8e4m3>::max();
    static constexpr fp8e4m3 min          = std::numeric_limits<fp8e4m3>::min();
    static constexpr fp8e4m3 denorm_min   = std::numeric_limits<fp8e4m3>::denorm_min();
    static constexpr fp8e4m3 infinity     = std::numeric_limits<fp8e4m3>::infinity();
    static constexpr fp8e4m3 low_extreme  = lowest;
    static constexpr fp8e4m3 high_extreme = max;
    static constexpr bool has_infinity    = false;
};

template <>
struct DtypeLimits<TOSA_REF_TYPE_FP8E5M2>
{
    static constexpr fp8e5m2 lowest       = std::numeric_limits<fp8e5m2>::lowest();
    static constexpr fp8e5m2 max          = std::numeric_limits<fp8e5m2>::max();
    static constexpr fp8e5m2 min          = std::numeric_limits<fp8e5m2>::min();
    static constexpr fp8e5m2 denorm_min   = std::numeric_limits<fp8e5m2>::denorm_min();
    static constexpr fp8e5m2 infinity     = std::numeric_limits<fp8e5m2>::infinity();
    static constexpr fp8e5m2 low_extreme  = -infinity;
    static constexpr fp8e5m2 high_extreme = infinity;
    static constexpr bool has_infinity    = true;
};

template <>
struct DtypeLimits<TOSA_REF_TYPE_INT32>
{
    static constexpr int32_t lowest       = std::numeric_limits<int32_t>::lowest();
    static constexpr int32_t max          = std::numeric_limits<int32_t>::max();
    static constexpr int32_t min          = std::numeric_limits<int32_t>::min();
    static constexpr int32_t denorm_min   = std::numeric_limits<int32_t>::denorm_min();
    static constexpr int32_t infinity     = std::numeric_limits<int32_t>::infinity();
    static constexpr int32_t low_extreme  = lowest;
    static constexpr int32_t high_extreme = max;
    static constexpr bool has_infinity    = false;
};

template <>
struct DtypeLimits<TOSA_REF_TYPE_INT16>
{
    static constexpr int16_t lowest       = std::numeric_limits<int16_t>::lowest();
    static constexpr int16_t max          = std::numeric_limits<int16_t>::max();
    static constexpr int16_t min          = std::numeric_limits<int16_t>::min();
    static constexpr int16_t denorm_min   = std::numeric_limits<int16_t>::denorm_min();
    static constexpr int16_t infinity     = std::numeric_limits<int16_t>::infinity();
    static constexpr int16_t low_extreme  = lowest;
    static constexpr int16_t high_extreme = max;
    static constexpr bool has_infinity    = false;
};

template <>
struct DtypeLimits<TOSA_REF_TYPE_INT8>
{
    static constexpr int8_t lowest       = std::numeric_limits<int8_t>::lowest();
    static constexpr int8_t max          = std::numeric_limits<int8_t>::max();
    static constexpr int8_t min          = std::numeric_limits<int8_t>::min();
    static constexpr int8_t denorm_min   = std::numeric_limits<int8_t>::denorm_min();
    static constexpr int8_t infinity     = std::numeric_limits<int8_t>::infinity();
    static constexpr int8_t low_extreme  = lowest;
    static constexpr int8_t high_extreme = max;
    static constexpr bool has_infinity   = false;
};

}    // namespace TosaReference

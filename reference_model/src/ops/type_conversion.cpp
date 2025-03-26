
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

#include "type_conversion.h"
#include "arith_util.h"
#include "cfloat.h"
#include "half.hpp"
#include "quant_util.h"
#include "template_types.h"
#include <cfenv>
#include <cmath>

using namespace TosaReference;
using namespace Eigen;
using namespace tosa;

template <int Rank, TOSA_REF_TYPE InDtype, TOSA_REF_TYPE OutDtype>
OpRescale<Rank, InDtype, OutDtype>::OpRescale(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_)
    : GraphNode(sgt_, Op_RESCALE, id_)
{
    setRequiredOperands(5, 1);
    setRequiredRank(0, 6);
    INIT_ATTRIBUTE(Rescale);

    QMax_s = getSignedMaximum<OutDtype>();
    QMin_s = getSignedMinimum<OutDtype>();
    QMax_u = getUnsignedMaximum<OutDtype>();
    QMin_u = getUnsignedMinimum<OutDtype>();
}

template <int Rank, TOSA_REF_TYPE InDtype, TOSA_REF_TYPE OutDtype>
OpRescale<Rank, InDtype, OutDtype>::~OpRescale()
{}

template <int Rank, TOSA_REF_TYPE InDtype, TOSA_REF_TYPE OutDtype>
int OpRescale<Rank, InDtype, OutDtype>::checkTensorAttributes()
{
    // Check Tosa Level
    auto tosa_level = g_func_config.tosa_level;
    LEVEL_CHECK(Rank <= tosa_level.MAX_RANK, "Rank should be smaller than or equal to MAX_RANK");

    if (validateRequiredOperands())
        return 1;

    if (validateRequiredRank(inputs[0]) && validateRequiredRank(inputs[1], 1, 1) &&
        validateRequiredRank(inputs[2], 1, 1) && validateRequiredRank(inputs[3], 1, 1) &&
        validateRequiredRank(inputs[4], 1, 1) && validateRequiredRank(outputs[0]))
        return 1;

    // output and input must be the same rank and size
    if (inputs[0]->matchRankSize(*outputs[0]))
    {
        printNodeValidationError("OpRescale: input and output rank/size must match");
        return 1;
    }

    in  = dynamic_cast<TosaReference::TensorTemplate<TIn>*>(inputs[0]);
    out = dynamic_cast<TosaReference::TensorTemplate<TOut>*>(outputs[0]);
    ASSERT_MEM(in && out);

    multiplierI32 = dynamic_cast<TosaReference::TensorTemplate<TMultiplierI32>*>(inputs[1]);
    multiplierI16 = dynamic_cast<TosaReference::TensorTemplate<TMultiplierI16>*>(inputs[1]);
    shift         = dynamic_cast<TosaReference::TensorTemplate<TShift>*>(inputs[2]);
    ASSERT_MEM(shift);

    in_zp  = dynamic_cast<TosaReference::TensorTemplate<TInZp>*>(inputs[3]);
    out_zp = dynamic_cast<TosaReference::TensorTemplate<TOutZp>*>(inputs[4]);
    ASSERT_MEM(in_zp && out_zp);

    if (attribute->scale32())
    {
        ASSERT_MEM(multiplierI32);
    }
    else
    {
        ASSERT_MEM(multiplierI16);
    }

    const bool input_unsigned  = attribute->input_unsigned();
    const bool output_unsigned = attribute->output_unsigned();
    const bool scale32         = attribute->scale32();
    const bool per_channel     = attribute->per_channel();
    const int in_rank          = inputs[0]->getRank();
    const auto rounding_mode   = attribute->rounding_mode();

    if (rounding_mode != RoundingMode_SINGLE_ROUND && rounding_mode != RoundingMode_DOUBLE_ROUND)
    {
        printNodeValidationError("OpRescale: Unsupported rounding mode");
        return 1;
    }

    if (scale32 && (InDtype == TOSA_REF_TYPE_INT48))
    {
        printNodeValidationError("OpRescale: Scale set to true but input type is INT48");
        return 1;
    }

    if (!scale32 && (rounding_mode == RoundingMode_DOUBLE_ROUND))
    {
        printNodeValidationError("OpRescale: Scale set to false but rounding mode set to double round");
        return 1;
    }

    if (input_unsigned && output_unsigned)
    {
        printNodeValidationError("OpRescale: input_unsigned and output_unsigned set to true");
        return 1;
    }

    if (isI32(OutDtype) && input_unsigned)
    {
        printNodeValidationError("OpRescale: Output signed int32 and input_unsigned set to true");
        return 1;
    }

    if (isI32(InDtype) && output_unsigned)
    {
        printNodeValidationError("OpRescale: Input signed int32 and output_unsigned set to true");
        return 1;
    }

    if (isI48(InDtype) && output_unsigned)
    {
        printNodeValidationError("OpRescale: Input signed int48 and output_unsigned set to true");
        return 1;
    }

    if (InDtype == TOSA_REF_TYPE_INT32 && input_unsigned)
    {
        printNodeValidationError("OpRescale: Input unsigned int32");
        return 1;
    }

    if (OutDtype == TOSA_REF_TYPE_INT32 && output_unsigned)
    {
        printNodeValidationError("OpRescale: Output unsigned int32");
        return 1;
    }

    if (InDtype == TOSA_REF_TYPE_INT48 && input_unsigned)
    {
        printNodeValidationError("OpRescale: Input unsigned int48");
        return 1;
    }

    if (per_channel && in_rank < 1)
    {
        printNodeValidationError("OpRescale: per_channel set to true and rank of input < 1");
        return 1;
    }

    return 0;
}

// helpers to convert types
static int64_t zero_extend(int8_t val)
{
    uint8_t* rval = reinterpret_cast<uint8_t*>(&val);
    return static_cast<int64_t>(*rval);
}
static int64_t zero_extend(int16_t val)
{
    uint16_t* rval = reinterpret_cast<uint16_t*>(&val);
    return static_cast<int64_t>(*rval);
}
static int64_t zero_extend(int32_t val)
{
    uint32_t* rval = reinterpret_cast<uint32_t*>(&val);
    return static_cast<int64_t>(*rval);
}
static int64_t sign_extend(uint8_t val)
{
    int8_t* rval = reinterpret_cast<int8_t*>(&val);
    return static_cast<int64_t>(*rval);
}
static int64_t sign_extend(uint16_t val)
{
    int16_t* rval = reinterpret_cast<int16_t*>(&val);
    return static_cast<int64_t>(*rval);
}

template <int Rank, TOSA_REF_TYPE InDtype, TOSA_REF_TYPE OutDtype>
int OpRescale<Rank, InDtype, OutDtype>::eval()
{
    const InEigenType input_zp   = this->in_zp->getTensor()(0);
    const OutEigenType output_zp = this->out_zp->getTensor()(0);
    std::vector<int32_t> multiplier;
    std::vector<int32_t> shift;
    const bool scale32       = attribute->scale32();
    const auto rounding_mode = attribute->rounding_mode();
    const bool double_round  = (rounding_mode == RoundingMode_DOUBLE_ROUND);
    const bool per_channel   = attribute->per_channel();

    // Note that how rescale op interprets signedness of the tensor depends on
    // the value of input_unsigned and output_unsigned attributes, and doesn't
    // care about the type of tensor itself.
    const bool input_unsigned  = attribute->input_unsigned();
    const bool output_unsigned = attribute->output_unsigned();

    auto value_extend64 = [=](InEigenType val, bool unsigned_val) -> int64_t {
        int64_t val_extended;
        switch (GetNumBits<InDtype>::value)
        {
            case 8:
                val_extended = (unsigned_val) ? zero_extend(static_cast<int8_t>(val))
                                              : sign_extend(static_cast<uint8_t>(val & 0xFF));
                break;
            case 16:
                val_extended = (unsigned_val) ? zero_extend(static_cast<int16_t>(val))
                                              : sign_extend(static_cast<uint16_t>(val & 0xFFFF));
                break;
            case 32:
                val_extended = (unsigned_val) ? zero_extend(static_cast<int32_t>(val)) : static_cast<int64_t>((val));
                break;
            case 48:
                if (unsigned_val)
                {
                    val_extended = static_cast<int64_t>(val) & 0x0000FFFFFFFFFFFF;
                }
                else
                {
                    // sign extend i48 to i64
                    val_extended = static_cast<int64_t>(val);
                    if (val_extended & 0x800000000000)
                    {
                        // in_val contains negative i48, sign extend to i64
                        val_extended |= 0xFFFF000000000000;
                    }
                }
                break;
            default:
                val_extended = static_cast<int64_t>(val);
                break;
        }
        return val_extended;
    };

    int64_t output_zp_extended = value_extend64(output_zp, output_unsigned);
    int64_t input_zp_extended  = value_extend64(input_zp, input_unsigned);

    // uint16 values can have zero_point 0 or 32768
    // int8/uint8 can have zero point within their valid range
    // No other types can have zero point != 0
    ERROR_IF(!isI8(InDtype) && !(isI16(InDtype) && input_unsigned) && (input_zp_extended != 0),
             "OpRescale: Input TOSA_REF_TYPE not INT8/UINT8/UINT16 and zero point not 0");
    ERROR_IF(!isI8(OutDtype) && !(isI16(OutDtype) && output_unsigned) && (output_zp_extended != 0),
             "OpRescale: Output TOSA_REF_TYPE not INT8/UINT8/UINT16 and zero point not 0");
    ERROR_IF(isI16(InDtype) && input_unsigned && (input_zp_extended != 0) && (input_zp_extended != 32768),
             "OpRescale: Input unsigned int16 and zero point (%ld) not 0 or 32768", input_zp_extended);
    ERROR_IF(isI16(OutDtype) && output_unsigned && (output_zp_extended != 0) && (output_zp_extended != 32768),
             "OpRescale: Output unsigned int16 and zero point (%ld) not 0 or 32768", output_zp_extended);

    // reshape [d0, d1, ..., dn] into [d0 * d1 ..., dn]
    Eigen::array<Eigen::Index, 2> shape_2d;
    shape_2d[0] = 1;
    if (Rank > 0)
    {
        for (size_t i = 0; i < Rank - 1; i++)
        {
            shape_2d[0] *= this->in->getShape()[i];
        }
        shape_2d[1] = this->in->getShape()[static_cast<size_t>(Rank - 1)];
    }
    else
    {
        shape_2d[1] = 1;
    }
    ETensor2<InEigenType> input_reshaped = this->in->getTensor().reshape(shape_2d);

    ETensor2<OutEigenType> output_2d(shape_2d);

    if (scale32)
    {
        auto multiplier_val = this->multiplierI32->getTensor();
        for (int i = 0; i < multiplier_val.size(); i++)
        {
            multiplier.push_back(static_cast<int32_t>(multiplier_val(i)));
        }
    }
    else
    {
        auto multiplier_val = this->multiplierI16->getTensor();
        for (int i = 0; i < multiplier_val.size(); i++)
        {
            multiplier.push_back(static_cast<int32_t>(multiplier_val(i)));
        }
    }
    auto shift_val = this->shift->getTensor();
    for (int i = 0; i < shift_val.size(); i++)
    {
        shift.push_back(static_cast<int32_t>(shift_val(i)));
    }

    if (per_channel)
    {
        ETensor2<InEigenType> curr_channel_slice_prescaled;
        ETensor2<OutEigenType> curr_channel_slice_postscaled;
        int32_t channel_multiplier, channel_shift;
        Eigen::array<Eigen::Index, 2> begin, size;
        size = Eigen::array<Eigen::Index, 2>({ shape_2d[0], 1 });
        try
        {
            for (int32_t i = 0; i < shape_2d[1]; i++)
            {
                begin                        = Eigen::array<Eigen::Index, 2>({ 0, i });
                curr_channel_slice_prescaled = input_reshaped.slice(begin, size);
                channel_multiplier           = multiplier[static_cast<size_t>(i)];
                channel_shift                = shift[static_cast<size_t>(i)];
                curr_channel_slice_postscaled =
                    curr_channel_slice_prescaled.unaryExpr([=](InEigenType in_val) -> OutEigenType {
                        int64_t input_zp_shifted = value_extend64(in_val, input_unsigned) - input_zp_extended;

                        int32_t scaled;
                        if (scale32)
                            scaled = TosaReference::QuantUtil::apply_scale_32(static_cast<int32_t>(input_zp_shifted),
                                                                              channel_multiplier, channel_shift,
                                                                              double_round);
                        else
                            scaled = TosaReference::QuantUtil::apply_scale_16(
                                input_zp_shifted, static_cast<int16_t>(channel_multiplier), channel_shift);

                        int64_t res_in_64     = static_cast<int64_t>(scaled) + output_zp_extended;
                        int64_t i32_max_in_64 = static_cast<int64_t>(std::numeric_limits<int32_t>::max());
                        int64_t i32_min_in_64 = static_cast<int64_t>(std::numeric_limits<int32_t>::min());

                        if (res_in_64 > i32_max_in_64 || res_in_64 < i32_min_in_64)
                        {
                            std::string desc = "scaling result [" + std::to_string(scaled) + "] plus output_zp [" +
                                               std::to_string(output_zp_extended) + "] not in i32 range";
                            throw desc;
                        }

                        // Treat the output values as unsigned if `output_unsigned` is true.
                        int32_t clipped_val = (output_unsigned)
                                                  ? applyClip<int32_t, uint32_t>(static_cast<int32_t>(res_in_64),
                                                                                 QMin_u, QMax_u, this->parent_sgt)
                                                  : applyClip<int32_t, int32_t>(static_cast<int32_t>(res_in_64), QMin_s,
                                                                                QMax_s, this->parent_sgt);

                        OutEigenType out_val = static_cast<OutEigenType>(clipped_val);
                        return out_val;
                    });

                for (int32_t j = 0; j < shape_2d[0]; j++)
                {
                    output_2d(j, i) = curr_channel_slice_postscaled(j, 0);
                }
            }
        }
        catch (std::string desc)
        {
            REQUIRE(false, "OpRescale failure: %s.", desc.c_str());
        }
    }
    else
    {
        int32_t tensor_multiplier = multiplier[0];
        int32_t tensor_shift      = shift[0];
        try
        {
            output_2d = input_reshaped.unaryExpr([=](InEigenType in_val) -> OutEigenType {
                int64_t input_zp_shifted = value_extend64(in_val, input_unsigned) - input_zp_extended;

                int32_t scaled;
                if (scale32)
                    scaled = TosaReference::QuantUtil::apply_scale_32(static_cast<int32_t>(input_zp_shifted),
                                                                      tensor_multiplier, tensor_shift, double_round);
                else
                    scaled = TosaReference::QuantUtil::apply_scale_16(
                        input_zp_shifted, static_cast<int16_t>(tensor_multiplier), tensor_shift);

                int64_t res_in_64     = static_cast<int64_t>(scaled) + output_zp_extended;
                int64_t i32_max_in_64 = IsSignedInt<OutDtype>()
                                            ? static_cast<int64_t>(std::numeric_limits<int32_t>::max())
                                            : static_cast<int64_t>(std::numeric_limits<uint32_t>::max());
                int64_t i32_min_in_64 = static_cast<int64_t>(std::numeric_limits<int32_t>::min());

                if (res_in_64 > i32_max_in_64 || res_in_64 < i32_min_in_64)
                {
                    std::string desc = "scaling result [" + std::to_string(scaled) + "] plus output_zp [" +
                                       std::to_string(output_zp_extended) + "] not in i32 range";
                    throw desc;
                }

                // Treat the output values as unsigned if `output_unsigned` is true.
                int32_t clipped_val = (output_unsigned) ? applyClip<int32_t, uint32_t>(static_cast<int32_t>(res_in_64),
                                                                                       QMin_u, QMax_u, this->parent_sgt)
                                                        : applyClip<int32_t, int32_t>(static_cast<int32_t>(res_in_64),
                                                                                      QMin_s, QMax_s, this->parent_sgt);

                OutEigenType out_val = static_cast<OutEigenType>(clipped_val);
                return out_val;
            });
        }
        catch (std::string desc)
        {
            REQUIRE(false, "OpRescale failure: %s.", desc.c_str());
        }
    }

    // reshape [d0 * d1 ..., dn] back to [d0, d1, ..., dn]
    Eigen::array<Eigen::Index, Rank> output_shape;
    for (size_t i = 0; i < Rank; i++)
    {
        output_shape[i] = this->out->getShape()[i];
    }
    this->out->getTensor() = output_2d.reshape(output_shape);

    return GraphNode::eval();
}

template <int Rank, TOSA_REF_TYPE InDtype, TOSA_REF_TYPE OutDtype>
OpCast<Rank, InDtype, OutDtype>::OpCast(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_)
    : GraphNode(sgt_, Op_CAST, id_)
{
    setRequiredOperands(1, 1);
    setRequiredRank(0, 6);
}

template <int Rank, TOSA_REF_TYPE InDtype, TOSA_REF_TYPE OutDtype>
OpCast<Rank, InDtype, OutDtype>::~OpCast()
{}

template <int Rank, TOSA_REF_TYPE InDtype, TOSA_REF_TYPE OutDtype>
int OpCast<Rank, InDtype, OutDtype>::checkTensorAttributes()
{
    // Check Tosa Level
    auto tosa_level = g_func_config.tosa_level;
    LEVEL_CHECK(Rank <= tosa_level.MAX_RANK, "Rank should be smaller than or equal to MAX_RANK");

    if (validateRequiredOperands())
        return 1;

    if (validateRequiredRank(inputs[0]) || validateRequiredRank(outputs[0]))
    {
        return 1;
    }

    // output and input must be the same rank and size
    if (inputs[0]->matchRankSize(*outputs[0]))
    {
        printNodeValidationError("OpCast: input and output rank/size must match");
        return 1;
    }

    in  = dynamic_cast<TosaReference::TensorTemplate<TIn>*>(inputs[0]);
    out = dynamic_cast<TosaReference::TensorTemplate<TOut>*>(outputs[0]);

    ASSERT_MEM(in && out);

    return 0;
}

template <int Rank, TOSA_REF_TYPE InDtype, TOSA_REF_TYPE OutDtype>
int OpCast<Rank, InDtype, OutDtype>::eval()
{
    this->out->getTensor() = this->in->getTensor().unaryExpr(cast_helper.get_fcn());

    return GraphNode::eval();
}

// Ensure rounding mode is reset after lambda execution
struct ScopedFEnv
{
    int old_mode;
    ScopedFEnv(int new_mode)
    {
        // Note: std::fegetround() can return a negative value
        // if the rounding mode is indeterminate.
        old_mode = std::fegetround();
        std::fesetround(new_mode);
    }
    ~ScopedFEnv()
    {
        // Only restore the original rounding mode
        // if it was successfully determined
        if (old_mode >= 0)
        {
            std::fesetround(old_mode);
        }
    }
};

template <TOSA_REF_TYPE InDtype, TOSA_REF_TYPE OutDtype>
CastHelper<InDtype, OutDtype>::CastHelper()
{
    constexpr int32_t outWidth = GetNumBits<OutDtype>().value;
    constexpr int32_t inWidth  = GetNumBits<InDtype>().value;

    fcn = [](InEigenType in) -> OutEigenType {
        if constexpr (std::is_integral_v<InEigenType> && std::is_integral_v<OutEigenType> && outWidth < inWidth)
        {
            // Truncate the value if it's outside the range of the output type.
            in = intTrunc<InEigenType, OutDtype>(in);
        }
        OutEigenType out = (OutEigenType)in;    // implicit sign_extend() if sizeof(out_t) >= sizeof(in_t)
        return out;
    };
}

template <TOSA_REF_TYPE InDtype>
CastHelper<InDtype, TOSA_REF_TYPE_BOOL>::CastHelper()
{
    fcn = [](InEigenType in) -> bool { return (in != 0) ? true : false; };
}

template <TOSA_REF_TYPE OutDtype>
CastHelper<TOSA_REF_TYPE_BOOL, OutDtype>::CastHelper()
{
    fcn = [](bool in) -> OutEigenType {
        OutEigenType out = in ? (OutEigenType)1 : (OutEigenType)0;
        return out;
    };
}

template <TOSA_REF_TYPE InDtype>
CastHelper<InDtype, TOSA_REF_TYPE_FP16>::CastHelper()
{
    // Integer data converted to fp16 (stored as fp32)
    fcn = [](InEigenType in) -> float {
        half_float::half h = half_float::half(static_cast<float>(in));
        float out          = half_float::half_cast<float, half_float::half>(h);
        return out;
    };
}

CastHelper<TOSA_REF_TYPE_FP32, TOSA_REF_TYPE_FP16>::CastHelper()
{
    // fp32 data converted to fp16 (stored as fp32)
    fcn = [](float in) -> float {
        float out = fpTrunc<TOSA_REF_TYPE_FP16>(in);    // truncate required for conversion from higher precision
        return out;
    };
}

template <TOSA_REF_TYPE InDtype>
CastHelper<InDtype, TOSA_REF_TYPE_BF16>::CastHelper()
{
    // Integer data converted to bf16 (stored as fp32)
    fcn = [](InEigenType in) -> float {
        float out = (float)in;    // default cast to float is round_to_nearest_float()
        return out;
    };
}

CastHelper<TOSA_REF_TYPE_FP32, TOSA_REF_TYPE_BF16>::CastHelper()
{
    // fp32 data converted to bf16 (stored as fp32)
    fcn = [](float in) -> float {
        return fpTrunc<TOSA_REF_TYPE_BF16>(in);    // truncate required for conversions from higher precision
    };
}

template <TOSA_REF_TYPE OutDtype>
CastHelper<TOSA_REF_TYPE_FP16, OutDtype>::CastHelper()
{
    // fp16 data (stored as fp32) converted to integer
    fcn = [](float in) -> OutEigenType {
        // Cast from float representation back to half_float before rounding
        half_float::half h = half_float::half(in);
        if (h >= half_float::half(float(OutMax)))
            return OutMax;

        if (h <= half_float::half(float(OutMin)))
            return OutMin;

        ScopedFEnv round_guard(FE_TONEAREST);
        h                = static_cast<half_float::half>(std::nearbyint(h));
        OutEigenType out = half_float::half_cast<OutEigenType, half_float::half>(h);

        return out;
    };
}

CastHelper<TOSA_REF_TYPE_FP16, TOSA_REF_TYPE_FP32>::CastHelper()
{
    // No-op since fp16 values treated internally as their fp32 representation
    fcn = [](float in) -> OutEigenType { return in; };
}

template <TOSA_REF_TYPE OutDtype>
CastHelper<TOSA_REF_TYPE_BF16, OutDtype>::CastHelper()
{
    // bf16 data (stored as fp32) converted to integer
    fcn = [](float in) -> OutEigenType {
        if (in >= float(OutMax))
            return OutMax;

        if (in <= float(OutMin))
            return OutMin;

        ScopedFEnv round_guard(FE_TONEAREST);
        OutEigenType out = static_cast<OutEigenType>(std::nearbyint(in));
        return out;
    };
}

CastHelper<TOSA_REF_TYPE_BF16, TOSA_REF_TYPE_FP32>::CastHelper()
{
    // No-op since bf16 values treated as truncated fp32 internally
    fcn = [](InEigenType in) -> OutEigenType { return in; };
}

template <TOSA_REF_TYPE InDtype>
CastHelper<InDtype, TOSA_REF_TYPE_FP32>::CastHelper()
{
    // Integer data converted to fp32
    fcn = [](InEigenType in) -> float {
        float out = (OutEigenType)in;    // default cast to float is round_to_nearest_float()
        return out;
    };
}

template <TOSA_REF_TYPE OutDtype>
CastHelper<TOSA_REF_TYPE_FP32, OutDtype>::CastHelper()
{
    // fp32 data converted to integer
    fcn = [](float in) -> OutEigenType {
        if (in >= float(OutMax))
            return OutMax;

        if (in <= float(OutMin))
            return OutMin;

        ScopedFEnv round_guard(FE_TONEAREST);
        OutEigenType out = static_cast<OutEigenType>(std::nearbyint(in));
        return out;
    };
}

template <TOSA_REF_TYPE OutDtype>
CastHelper<TOSA_REF_TYPE_FP8E4M3, OutDtype>::CastHelper()
{
    // fp8e4m3 data (stored as fp32) converted to integer
    fcn = [](float in) -> OutEigenType {
        if (in >= float(OutMax))
            return OutMax;
        if (in <= float(OutMin))
            return OutMin;

        ScopedFEnv round_guard(FE_TONEAREST);
        OutEigenType out = static_cast<OutEigenType>(std::nearbyint(in));
        return out;
    };
}

CastHelper<TOSA_REF_TYPE_FP8E4M3, TOSA_REF_TYPE_FP16>::CastHelper()
{
    // fp8e4m3 data (stored as fp32) converted to fp16 (stored as fp32)
    fcn = [](float in) -> float {
        half_float::half h = half_float::half(in);
        float out          = half_float::half_cast<half_float::half, float>(h);
        return out;
    };
}

CastHelper<TOSA_REF_TYPE_FP8E4M3, TOSA_REF_TYPE_BF16>::CastHelper()
{
    // fp8e4m3 data (stored as fp32) converted to bf16 (stored as fp32)
    fcn = [](float in) -> float { return (float)in; };
}

CastHelper<TOSA_REF_TYPE_FP8E4M3, TOSA_REF_TYPE_FP32>::CastHelper()
{
    // fp8e4m3 data (stored as fp32) converted to fp32
    fcn = [](InEigenType in) -> OutEigenType { return in; };
}

CastHelper<TOSA_REF_TYPE_FP8E4M3, TOSA_REF_TYPE_FP64>::CastHelper()
{
    // fp8e4m3 data (stored as fp32) converted to fp64
    fcn = [](float in) -> double { return static_cast<double>(in); };
}

template <TOSA_REF_TYPE OutDtype>
CastHelper<TOSA_REF_TYPE_FP8E5M2, OutDtype>::CastHelper()
{
    // fp8e5m2 data (stored as fp32) converted to integer
    fcn = [](float in) -> OutEigenType {
        if (in >= float(OutMax))
            return OutMax;
        if (in <= float(OutMin))
            return OutMin;

        ScopedFEnv round_guard(FE_TONEAREST);
        OutEigenType out = static_cast<OutEigenType>(std::nearbyint(in));
        return out;
    };
}

CastHelper<TOSA_REF_TYPE_FP8E5M2, TOSA_REF_TYPE_FP16>::CastHelper()
{
    // fp8e5m2 data (stored as fp32) converted to fp16 (stored as fp32)
    fcn = [](float in) -> float {
        half_float::half h = half_float::half(in);
        float out          = half_float::half_cast<half_float::half, float>(h);
        return out;
    };
}

CastHelper<TOSA_REF_TYPE_FP8E5M2, TOSA_REF_TYPE_BF16>::CastHelper()
{
    // fp8e5m2 data (stored as fp32) converted to bf16 (stored as fp32)
    fcn = [](float in) -> float { return (float)in; };
}

CastHelper<TOSA_REF_TYPE_FP8E5M2, TOSA_REF_TYPE_FP32>::CastHelper()
{
    // fp8e5m2 data (stored as fp32) converted to fp32
    fcn = [](InEigenType in) -> OutEigenType { return in; };
}

CastHelper<TOSA_REF_TYPE_FP8E5M2, TOSA_REF_TYPE_FP64>::CastHelper()
{
    // fp8e5m2 data (stored as fp32) converted to fp64
    fcn = [](float in) -> double { return static_cast<double>(in); };
}

template <TOSA_REF_TYPE InDtype>
CastHelper<InDtype, TOSA_REF_TYPE_FP8E4M3>::CastHelper()
{
    // Integer data converted to fp8e4m3 (stored as fp32)
    fcn = [](InEigenType in) -> float {
        auto f    = static_cast<fp8e4m3>(float(in));
        float out = static_cast<float>(f);
        return out;
    };
}

CastHelper<TOSA_REF_TYPE_FP16, TOSA_REF_TYPE_FP8E4M3>::CastHelper()
{
    // fp16 data (stored as fp32) converted to fp8e4m3 (stored as fp32)
    fcn = [](float in) -> float { return static_cast<float>(static_cast<fp8e4m3>(in)); };
}

CastHelper<TOSA_REF_TYPE_BF16, TOSA_REF_TYPE_FP8E4M3>::CastHelper()
{
    // bf16 data (stored as fp32) converted to fp8e4m3 (stored as fp32)
    fcn = [](float in) -> float { return static_cast<float>(static_cast<fp8e4m3>(in)); };
}

CastHelper<TOSA_REF_TYPE_FP32, TOSA_REF_TYPE_FP8E4M3>::CastHelper()
{
    // fp32 data converted to fp8e4m3 (stored as fp32)
    fcn = [](float in) -> float { return static_cast<float>(static_cast<fp8e4m3>(in)); };
}

template <TOSA_REF_TYPE InDtype>
CastHelper<InDtype, TOSA_REF_TYPE_FP8E5M2>::CastHelper()
{
    // Integer data converted to fp8e5m2 (stored as fp32)
    fcn = [](InEigenType in) -> float { return static_cast<float>(static_cast<fp8e5m2>(in)); };
}

CastHelper<TOSA_REF_TYPE_FP16, TOSA_REF_TYPE_FP8E5M2>::CastHelper()
{
    // fp16 data (stored as fp32) converted to fp8e5m2 (stored as fp32)
    fcn = [](float in) -> float { return static_cast<float>(static_cast<fp8e5m2>(in)); };
}

CastHelper<TOSA_REF_TYPE_BF16, TOSA_REF_TYPE_FP8E5M2>::CastHelper()
{
    // bf16 data (stored as fp32) converted to fp8e5m2 (stored as fp32)
    fcn = [](float in) -> float { return static_cast<float>(static_cast<fp8e5m2>(in)); };
}

CastHelper<TOSA_REF_TYPE_FP32, TOSA_REF_TYPE_FP8E5M2>::CastHelper()
{
    // fp32 data converted to fp8e5m2 (stored as fp32)
    fcn = [](float in) -> float { return static_cast<float>(static_cast<fp8e5m2>(in)); };
}

CastHelper<TOSA_REF_TYPE_FP64, TOSA_REF_TYPE_FP8E4M3>::CastHelper()
{
    // fp64 data converted to fp8e5m2 (stored as fp32)
    fcn = [](double in) -> float {
        float out = static_cast<float>(ct::compat::cast<fp8e4m3>(in));
        return out;
    };
}

CastHelper<TOSA_REF_TYPE_FP64, TOSA_REF_TYPE_FP8E5M2>::CastHelper()
{
    // fp64 data converted to fp8e5m2 (stored as fp32)
    fcn = [](double in) -> float {
        float out = static_cast<float>(ct::compat::cast<fp8e5m2>(in));
        return out;
    };
}

template <TOSA_REF_TYPE OutDtype>
CastHelper<TOSA_REF_TYPE_FP64, OutDtype>::CastHelper()
{
    switch (OutDtype)
    {
        case TOSA_REF_TYPE_INT8:
        case TOSA_REF_TYPE_INT16:
        case TOSA_REF_TYPE_INT32:
            // fp64 data converted to integer
            fcn = [](InEigenType in) -> OutEigenType {
                if (in >= double(OutMax))
                    return OutMax;

                if (in <= double(OutMin))
                    return OutMin;

                ScopedFEnv round_guard(FE_TONEAREST);
                OutEigenType out = static_cast<OutEigenType>(std::nearbyint(in));
                return out;
            };
            break;
        case TOSA_REF_TYPE_FP64:
            // no op
            fcn = [](InEigenType in) -> OutEigenType { return static_cast<OutEigenType>(in); };
            break;
        default:
            ASSERT_MSG(false, "unsupported TOSA_REF_TYPE %s", EnumNameTOSAREFTYPE(OutDtype));
    }
}

// template explicit instantiation
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, BOOL, INT8);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, BOOL, INT16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, BOOL, INT32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, INT8, BOOL);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, INT8, INT16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, INT8, INT32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, INT8, FP16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, INT8, BF16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, INT8, FP32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, INT16, BOOL);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, INT16, INT8);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, INT16, INT32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, INT16, FP16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, INT16, BF16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, INT16, FP32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, INT32, BOOL);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, INT32, INT8);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, INT32, INT16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, INT32, FP16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, INT32, BF16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, INT32, FP32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, FP16, INT8);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, FP16, INT16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, FP16, INT32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, FP16, FP32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, BF16, INT8);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, BF16, INT16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, BF16, INT32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, BF16, FP32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, FP32, INT8);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, FP32, INT16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, FP32, INT32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, FP32, FP16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, FP32, BF16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, FP64, INT8);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, FP64, INT16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, FP64, INT32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, FP64, FP8E4M3);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, FP64, FP8E5M2);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, FP64, FP64);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, INT8, FP64);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, INT16, FP64);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, INT32, FP64);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, BF16, FP8E4M3);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, BF16, FP8E5M2);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, FP8E4M3, FP16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, FP8E4M3, BF16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, FP8E4M3, FP32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, FP8E4M3, FP64);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, FP8E5M2, FP16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, FP8E5M2, BF16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, FP8E5M2, FP32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, FP8E5M2, FP64);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, FP16, FP8E4M3);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, FP16, FP8E5M2);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, FP32, FP8E4M3);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, FP32, FP8E5M2);

DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, INT8, INT8);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, INT8, INT16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, INT8, INT32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, INT16, INT8);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, INT16, INT16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, INT16, INT32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, INT32, INT8);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, INT32, INT16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, INT32, INT32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, INT48, INT8);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, INT48, INT16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, INT48, INT32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, UINT8, INT8);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, UINT8, INT16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, UINT8, INT32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, UINT16, INT8);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, UINT16, INT16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, UINT16, INT32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, INT8, UINT8);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, INT8, UINT16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, INT16, UINT8);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, INT16, UINT16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, INT32, UINT8);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, INT32, UINT16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, INT48, UINT8);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, INT48, UINT16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, UINT8, UINT8);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, UINT8, UINT16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, UINT16, UINT8);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, UINT16, UINT16);

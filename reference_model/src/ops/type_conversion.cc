
// Copyright (c) 2020-2022, ARM Limited.
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
#include "quant_util.h"
#include "arith_util.h"
#include "template_types.h"
#include <cmath>
#include "half.hpp"

using namespace TosaReference;
using namespace Eigen;
using namespace tosa;

template <int Rank, DType InDtype, DType OutDtype>
OpRescale<Rank, InDtype, OutDtype>::OpRescale(SubgraphTraverser* sgt_,
                                              TosaAttributeBase* attribute_,
                                              uint64_t id_)
    : GraphNode(sgt_, Op_RESCALE, id_)
{
    setRequiredOperands(1, 1);
    setRequiredRank(0, 6);
    INIT_ATTRIBUTE(Rescale);
}

template <int Rank, DType InDtype, DType OutDtype>
OpRescale<Rank, InDtype, OutDtype>::~OpRescale()
{
    if (attribute)
        delete attribute;
}

template <int Rank, DType InDtype, DType OutDtype>
int OpRescale<Rank, InDtype, OutDtype>::checkTensorAttributes()
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
        printNodeValidationError("OpRescale: input and output rank/size must match");
        return 1;
    }

    in  = dynamic_cast<TosaReference::TensorTemplate<TIn>*>(inputs[0]);
    out = dynamic_cast<TosaReference::TensorTemplate<TOut>*>(outputs[0]);

    ASSERT_MEM(in && out);

    if ((InDtype != DType_INT8) && (InDtype != DType_UINT8) && (InDtype != DType_UINT16) && (attribute->input_zp() != 0))
    {
        printNodeValidationError("OpRescale: Input DType not INT8/UINT8/UINT16 and zero point not 0");
        return 1;
    }

    if ((OutDtype != DType_INT8) && (OutDtype != DType_UINT8) && (OutDtype != DType_UINT16) && (attribute->output_zp() != 0))
    {
        printNodeValidationError("OpRescale: Output DType not INT8/UINT8/UINT16 and zero point not 0");
        return 1;
    }

    if ((InDtype == DType_UINT16) && ((attribute->input_zp() != 0) && (attribute->input_zp() != 32768)))
    {
        printNodeValidationError("OpRescale: Input DType UINT16 and zero point not 0 or 32768");
        return 1;
    }

    if ((OutDtype == DType_UINT16) && ((attribute->output_zp() != 0) && (attribute->output_zp() != 32768)))
    {
        printNodeValidationError("OpRescale: Output DType UINT16 and zero point not 0 or 32768");
        return 1;
    }

    if (attribute->scale32() && (InDtype == DType_INT48))
    {
        printNodeValidationError("OpRescale: Scale set to true but input type is INT48");
        return 1;
    }

    if ((!attribute->scale32()) && attribute->double_round())
    {
        printNodeValidationError("OpRescale: Scale set to false but double round set to true");
        return 1;
    }

    return 0;
}

template <int Rank, DType InDtype, DType OutDtype>
int OpRescale<Rank, InDtype, OutDtype>::eval()
{
    int32_t input_zp                = attribute->input_zp();
    int32_t output_zp               = attribute->output_zp();
    std::vector<int32_t> multiplier = attribute->multiplier();
    std::vector<int32_t> shift      = attribute->shift();
    bool scale32                    = attribute->scale32();
    bool double_round               = attribute->double_round();
    bool per_channel                = attribute->per_channel();

    // reshape [d0, d1, ..., dn] into [d0 * d1 ..., dn]
    Eigen::array<Eigen::Index, 2> shape_2d;
    shape_2d[0] = 1;
    if (Rank > 0)
    {
        for (int i = 0; i < Rank - 1; i++)
        {
            shape_2d[0] *= this->in->getShape()[i];
        }
        shape_2d[1] = this->in->getShape()[Rank - 1];
    }
    else
    {
        shape_2d[1] = 1;
    }
    ETensor2<InEigenType> input_reshaped = this->in->getTensor().reshape(shape_2d);

    ETensor2<OutEigenType> output_2d(shape_2d);

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
                channel_multiplier           = multiplier[i];
                channel_shift                = shift[i];
                curr_channel_slice_postscaled =
                    curr_channel_slice_prescaled.unaryExpr([input_zp, output_zp, channel_multiplier, channel_shift,
                                                            double_round, scale32](InEigenType in_val) -> OutEigenType {
                        InEigenType input_zp_shifted = in_val - (InEigenType)input_zp;
                        int32_t scaled;
                        if (scale32)
                            scaled = TosaReference::QuantUtil::apply_scale_32(input_zp_shifted, channel_multiplier,
                                                                              channel_shift, double_round);
                        else
                            scaled = TosaReference::QuantUtil::apply_scale_16(input_zp_shifted, channel_multiplier,
                                                                              channel_shift);
                        int64_t res_in_64 = static_cast<int64_t>(scaled) + output_zp;
                        int64_t i32_max_in_64 = static_cast<int64_t>(std::numeric_limits<int32_t>::max());
                        int64_t i32_min_in_64 = static_cast<int64_t>(std::numeric_limits<int32_t>::min());
                        if (res_in_64 > i32_max_in_64 || res_in_64 < i32_min_in_64)
                        {
                            std::string desc = "scaling result [" + std::to_string(scaled) + "] plus output_zp [" + std::to_string(output_zp) + "] not in i32 range";
                            throw desc;
                        }
                        OutEigenType out_val = static_cast<OutEigenType>(res_in_64);
                        out_val              = std::max<OutEigenType>(out_val, QMin);
                        out_val              = std::min<OutEigenType>(out_val, QMax);
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
            output_2d = input_reshaped.unaryExpr([input_zp, output_zp, tensor_multiplier, tensor_shift, double_round,
                                                  scale32](InEigenType in_val) -> OutEigenType {
                InEigenType input_zp_shifted = in_val - (InEigenType)input_zp;
                int32_t scaled;
                if (scale32)
                    scaled = TosaReference::QuantUtil::apply_scale_32(input_zp_shifted, tensor_multiplier, tensor_shift,
                                                                      double_round);
                else
                    scaled =
                        TosaReference::QuantUtil::apply_scale_16(input_zp_shifted, tensor_multiplier, tensor_shift);
                int64_t res_in_64 = static_cast<int64_t>(scaled) + output_zp;
                int64_t i32_max_in_64 = static_cast<int64_t>(std::numeric_limits<int32_t>::max());
                int64_t i32_min_in_64 = static_cast<int64_t>(std::numeric_limits<int32_t>::min());
                if (res_in_64 > i32_max_in_64 || res_in_64 < i32_min_in_64)
                {
                    std::string desc = "scaling result [" + std::to_string(scaled) + "] plus output_zp [" + std::to_string(output_zp) + "] not in i32 range";
                    throw desc;
                }

                OutEigenType out_val = static_cast<OutEigenType>(res_in_64);
                out_val              = std::max<OutEigenType>(out_val, QMin);
                out_val              = std::min<OutEigenType>(out_val, QMax);
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
    for (int i = 0; i < Rank; i++)
    {
        output_shape[i] = this->out->getShape()[i];
    }
    this->out->getTensor() = output_2d.reshape(output_shape);

    return GraphNode::eval();
}

template <int Rank, DType InDtype, DType OutDtype>
OpCast<Rank, InDtype, OutDtype>::OpCast(SubgraphTraverser* sgt_,
                                        TosaAttributeBase* attribute_,
                                        uint64_t id_)
    : GraphNode(sgt_, Op_CAST, id_)
{
    setRequiredOperands(1, 1);
    setRequiredRank(0, 6);
}

template <int Rank, DType InDtype, DType OutDtype>
OpCast<Rank, InDtype, OutDtype>::~OpCast()
{}

template <int Rank, DType InDtype, DType OutDtype>
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

template <int Rank, DType InDtype, DType OutDtype>
int OpCast<Rank, InDtype, OutDtype>::eval()
{
    this->out->getTensor() = this->in->getTensor().unaryExpr(cast_helper.get_fcn());

    return GraphNode::eval();
}

template <DType InDtype, DType OutDtype>
CastHelper<InDtype, OutDtype>::CastHelper()
{
    fcn = [](InEigenType in) -> OutEigenType {
        OutEigenType out = (OutEigenType)in;    // implicit sign_extend() if sizeof(out_t) >= sizeof(in_t)
        return out;
    };
}

template <DType InDtype>
CastHelper<InDtype, DType_BOOL>::CastHelper()
{
    fcn = [](InEigenType in) -> bool { return (in != 0) ? true : false; };
}

template <DType OutDtype>
CastHelper<DType_BOOL, OutDtype>::CastHelper()
{
    fcn = [](bool in) -> OutEigenType {
        OutEigenType out = in ? (OutEigenType)1 : (OutEigenType)0;
        return out;
    };
}

template <DType InDtype>
CastHelper<InDtype, DType_FP16>::CastHelper()
{
    // Integer data converted to fp16 (stored as fp32)
    fcn = [](InEigenType in) -> float {
        half_float::half h = half_float::half(in);
        float out = half_float::half_cast<float, half_float::half>(h);
        return out;
    };
}

CastHelper<DType_FP32, DType_FP16>::CastHelper()
{
    // fp32 data converted to fp16 (stored as fp32)
    fcn = [](float in) -> float {
        float out = fpTrunc<DType_FP16>(in);  // truncate required for conversion from higher precision
        return out;
    };
}

template <DType InDtype>
CastHelper<InDtype, DType_BF16>::CastHelper()
{
    // Integer data converted to bf16 (stored as fp32)
    fcn = [](InEigenType in) -> float {
        float out = (float)in;    // default cast to float is round_to_nearest_float()
        return out;
    };
}

CastHelper<DType_FP32, DType_BF16>::CastHelper()
{
    // fp32 data converted to bf16 (stored as fp32)
    fcn = [](float in) -> float {
        return fpTrunc<DType_BF16>(in);  // truncate required for conversions from higher precision
    };
}

template <DType OutDtype>
CastHelper<DType_FP16, OutDtype>::CastHelper()
{
    // fp16 data (stored as fp32) converted to integer
    fcn = [](float in) -> OutEigenType {
        // Cast from float representation back to half_float before rounding
        half_float::half h = half_float::half(in);
        h = std::rint(h);
        OutEigenType out = half_float::half_cast<OutEigenType, half_float::half>(h);
        out              = std::max<OutEigenType>(out, OutMin);
        out              = std::min<OutEigenType>(out, OutMax);
        return out;
    };
}

CastHelper<DType_FP16, DType_FP32>::CastHelper()
{
    // No-op since fp16 values treated internally as their fp32 representation
    fcn = [](float in) -> OutEigenType {
        return in;
    };
}

template <DType OutDtype>
CastHelper<DType_BF16, OutDtype>::CastHelper()
{
    // bf16 data (stored as fp32) converted to integer
    fcn = [](float in) -> OutEigenType {
        OutEigenType out = std::round(in);
        out              = std::max<OutEigenType>(out, OutMin);
        out              = std::min<OutEigenType>(out, OutMax);
        return out;
    };
}

CastHelper<DType_BF16, DType_FP32>::CastHelper()
{
    // No-op since bf16 values treated as truncated fp32 internally
    fcn = [](InEigenType in) -> OutEigenType {
        return in;
    };
}

template <DType InDtype>
CastHelper<InDtype, DType_FP32>::CastHelper()
{
    // Integer data converted to fp32
    fcn = [](InEigenType in) -> float {
        float out = (OutEigenType)in;    // default cast to float is round_to_nearest_float()
        return out;
    };
}

template <DType OutDtype>
CastHelper<DType_FP32, OutDtype>::CastHelper()
{
    // fp32 data converted to integer
    fcn = [](float in) -> OutEigenType {
        OutEigenType out = std::rint(in);
        out              = std::max<OutEigenType>(out, OutMin);
        out              = std::min<OutEigenType>(out, OutMax);
        return out;
    };
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
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, UINT16, INT16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, INT8, UINT8);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, INT16, UINT8);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, INT16, UINT16);

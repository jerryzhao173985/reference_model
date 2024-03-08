
// Copyright (c) 2020-2024, ARM Limited.
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

#include "activation_funcs.h"
#include "arith_util.h"
#include "quant_util.h"
#include "template_types.h"
#include "tosa_serialization_handler.h"
#include <cmath>

using namespace TosaReference;
using namespace Eigen;
using namespace tosa;

template <int Rank, TOSA_REF_TYPE Dtype>
int OpClamp<Rank, Dtype>::register_fcn()
{
    // Check Tosa Level
    auto tosa_level = g_func_config.tosa_level;
    LEVEL_CHECK(Rank <= tosa_level.MAX_RANK, "Rank should be smaller than or equal to MAX_RANK");

    switch (Dtype)
    {
        case TOSA_REF_TYPE_FP16:
        case TOSA_REF_TYPE_BF16:
        case TOSA_REF_TYPE_FP32: {
            std::vector<float> min_float_data, max_float_data;
            TosaSerializationHandler::ConvertU8toF32(attribute->min_val(), /* size = */ 1, min_float_data);
            TosaSerializationHandler::ConvertU8toF32(attribute->max_val(), /* size = */ 1, max_float_data);
            InEigenType min = (InEigenType)min_float_data[0];
            InEigenType max = (InEigenType)max_float_data[0];
            ERROR_IF(max < min, "OpClamp: max smaller than min");

            this->fcn = [min, max](InEigenType a) -> OutEigenType {
                return fpTrunc<Dtype>(a <= min ? min : a >= max ? max : a);
            };
        }
        break;
        case TOSA_REF_TYPE_FP64: {
            std::vector<float> min_float_data, max_float_data;
            TosaSerializationHandler::ConvertU8toF32(attribute->min_val(), /* size = */ 1, min_float_data);
            TosaSerializationHandler::ConvertU8toF32(attribute->max_val(), /* size = */ 1, max_float_data);
            InEigenType min = (InEigenType)min_float_data[0];
            InEigenType max = (InEigenType)max_float_data[0];
            ERROR_IF(max < min, "OpClamp: max smaller than min");

            this->fcn = [min, max](InEigenType a) -> OutEigenType { return (a <= min ? min : a >= max ? max : a); };
        }
        break;
        case TOSA_REF_TYPE_INT8: {
            std::vector<int32_t> min_int_data, max_int_data;
            TosaSerializationHandler::ConvertU8toI32(attribute->min_val(), /* size = */ 1, min_int_data);
            TosaSerializationHandler::ConvertU8toI32(attribute->max_val(), /* size = */ 1, max_int_data);
            int8_t min = (int8_t)min_int_data[0];
            int8_t max = (int8_t)max_int_data[0];

            ERROR_IF(max < min, "OpClamp: max smaller than min");
            this->fcn = [min, max](int8_t a) -> int8_t { return a <= min ? min : a >= max ? max : a; };
        }
        case TOSA_REF_TYPE_INT16: {
            std::vector<int32_t> min_int_data, max_int_data;
            TosaSerializationHandler::ConvertU8toI32(attribute->min_val(), /* size = */ 1, min_int_data);
            TosaSerializationHandler::ConvertU8toI32(attribute->max_val(), /* size = */ 1, max_int_data);
            int16_t min = (int16_t)min_int_data[0];
            int16_t max = (int16_t)max_int_data[0];

            ERROR_IF(max < min, "OpClamp: max smaller than min");
            this->fcn = [min, max](int16_t a) -> int16_t { return a <= min ? min : a >= max ? max : a; };
        }
        break;
        default:
            ERROR_IF(true, "unsupported TOSA_REF_TYPE %s", EnumNameTOSAREFTYPE(Dtype));
    }

    return 0;
}

template <int Rank, TOSA_REF_TYPE Dtype>
OpClamp<Rank, Dtype>::~OpClamp()
{
    if (attribute)
        delete attribute;
}

template <int Rank, TOSA_REF_TYPE Dtype>
int OpSigmoid<Rank, Dtype>::register_fcn()
{
    // Check Tosa Level
    auto tosa_level = g_func_config.tosa_level;
    LEVEL_CHECK(Rank <= tosa_level.MAX_RANK, "Rank should be smaller than or equal to MAX_RANK");

    switch (Dtype)
    {
        case TOSA_REF_TYPE_FP16:
        case TOSA_REF_TYPE_BF16:
        case TOSA_REF_TYPE_FP32:
            this->fcn = [](InEigenType a) -> OutEigenType { return fpTrunc<Dtype>(1.f / (1.f + (expf(-1.f * a)))); };
            break;
        case TOSA_REF_TYPE_FP64:
            if (g_func_config.abs_mode)
            {
                // ABS_ERROR bounds return 2*(1+abs(a))
                this->fcn = [](InEigenType a) -> OutEigenType { return 2.0 * (1.0 + (a > (InEigenType)0 ? a : (-a))); };
            }
            else
            {
                this->fcn = [](InEigenType a) -> OutEigenType { return (1.L / (1.L + (exp(-1.L * a)))); };
            }
            break;
        default:
            ERROR_IF(true, "unsupported TOSA_REF_TYPE %s", EnumNameTOSAREFTYPE(Dtype));
    }

    return 0;
}

template <int Rank, TOSA_REF_TYPE Dtype>
int OpTanh<Rank, Dtype>::register_fcn()
{
    // Check Tosa Level
    auto tosa_level = g_func_config.tosa_level;
    LEVEL_CHECK(Rank <= tosa_level.MAX_RANK, "Rank should be smaller than or equal to MAX_RANK");

    switch (Dtype)
    {
        case TOSA_REF_TYPE_FP16:
        case TOSA_REF_TYPE_BF16:
        case TOSA_REF_TYPE_FP32:
            this->fcn = [](InEigenType a) -> OutEigenType { return fpTrunc<Dtype>(tanhf(a)); };
            break;
        case TOSA_REF_TYPE_FP64:
            if (g_func_config.abs_mode)
            {
                // ABS_ERROR bounds return 4*(1+abs(a))
                this->fcn = [](InEigenType a) -> OutEigenType { return 4.0 * (1.0 + (a > (InEigenType)0 ? a : (-a))); };
            }
            else
            {
                this->fcn = [](InEigenType a) -> OutEigenType { return tanh(a); };
            }
            break;
        default:
            ERROR_IF(true, "unsupported TOSA_REF_TYPE %s", EnumNameTOSAREFTYPE(Dtype));
    }

    return 0;
}

template <int Rank, TOSA_REF_TYPE Dtype>
int OpErf<Rank, Dtype>::register_fcn()
{
    // Check Tosa Level
    auto tosa_level = g_func_config.tosa_level;
    LEVEL_CHECK(Rank <= tosa_level.MAX_RANK, "Rank should be similar than or equal to MAX_RANK");

    switch (Dtype)
    {
        case TOSA_REF_TYPE_FP16:
        case TOSA_REF_TYPE_BF16:
        case TOSA_REF_TYPE_FP32:
            this->fcn = [](InEigenType a) -> OutEigenType { return fpTrunc<Dtype>(erff(a)); };
            break;
        case TOSA_REF_TYPE_FP64:
            this->fcn = [](InEigenType a) -> OutEigenType { return erf(a); };
            break;
        default:
            ERROR_IF(true, "unsupported TOSA_REF_TYPE %s", EnumNameTOSAREFTYPE(Dtype));
    }

    return 0;
}

// template explicit instantiation
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpClamp, FP16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpClamp, BF16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpClamp, FP32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpClamp, INT8);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpClamp, INT16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpClamp, FP64);

DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpSigmoid, BF16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpSigmoid, FP16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpSigmoid, FP32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpSigmoid, FP64);

DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpTanh, BF16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpTanh, FP16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpTanh, FP32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpTanh, FP64);

DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpErf, BF16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpErf, FP16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpErf, FP32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpErf, FP64);

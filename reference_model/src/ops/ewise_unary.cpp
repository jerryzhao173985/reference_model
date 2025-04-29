
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

#include "ewise_unary.h"
#include "quant_util.h"
#include "template_types.h"
#include <cmath>

using namespace TosaReference;
using namespace Eigen;
using namespace tosa;

template <int Rank, TOSA_REF_TYPE Dtype>
UnaryNode<Rank, Dtype>::UnaryNode(SubgraphTraverser* sgt_, const Op& op_, uint64_t id_)
    : GraphNode(sgt_, op_, id_)
{
    setRequiredOperands(1, 1);

    fcn = [](InEigenType a) -> OutEigenType {
        ASSERT_MSG(0, "In default UnaryNode function, missing function registration");
        return OutEigenType();
    };
}

template <int Rank, TOSA_REF_TYPE Dtype>
UnaryNode<Rank, Dtype>::~UnaryNode()
{}

template <int Rank, TOSA_REF_TYPE Dtype>
int UnaryNode<Rank, Dtype>::checkTensorAttributes()
{
    // Check Tosa Level
    auto tosa_level = g_func_config.tosa_level;
    LEVEL_CHECK(Rank <= tosa_level.MAX_RANK, "Rank should be smaller than or equal to MAX_RANK");

    if (validateRequiredOperands())
        return 1;

    // output and input must be the same types
    if (inputs[0]->matchRankTypeShape(*outputs[0]))
    {
        printNodeValidationError("UnaryNode: input and output rank/type/shape must match");
        return 1;
    }

    a      = dynamic_cast<TosaReference::TensorTemplate<TIn>*>(inputs[0]);
    result = dynamic_cast<TosaReference::TensorTemplate<TOut>*>(outputs[0]);

    ASSERT_MEM(a && result);

    return 0;
}

template <int Rank, TOSA_REF_TYPE Dtype>
int UnaryNode<Rank, Dtype>::eval()
{
    // call register_fcn() here to ensure inputs/outputs have been connected
    // to the node by the time register_fcn() is called for Clamp Operator
    if (register_fcn())
    {
        return 1;
    }

    this->result->getTensor() = this->a->getTensor().unaryExpr(this->fcn);

    if (this->parent_sgt->getGraphStatus() == GraphStatus::TOSA_ERROR)
    {
        return 1;
    }

    return GraphNode::eval();
}

template <int Rank, TOSA_REF_TYPE Dtype>
int OpAbs<Rank, Dtype>::register_fcn()
{
    switch (Dtype)
    {
        case TOSA_REF_TYPE_FP32:    // No fpTrunc for FP32 as it is a no-op
        case TOSA_REF_TYPE_FP64:
            this->fcn = [](InEigenType a) -> OutEigenType { return a > (InEigenType)0 ? a : (-a); };
            break;
        case TOSA_REF_TYPE_INT32:
            this->fcn = [this](InEigenType a) -> OutEigenType {
                int64_t res_in_64     = a > (InEigenType)0 ? static_cast<int64_t>(a) : -static_cast<int64_t>(a);
                int64_t i32_max_in_64 = static_cast<int64_t>(std::numeric_limits<int32_t>::max());
                REQUIRE(res_in_64 <= i32_max_in_64, "OpAbs: result not in acc type range (int32)");
                return static_cast<OutEigenType>(res_in_64);
            };
            break;
        case TOSA_REF_TYPE_FP16:
        case TOSA_REF_TYPE_BF16:
            this->fcn = [](InEigenType a) -> OutEigenType {
                return static_cast<OutEigenType>(fpTrunc<Dtype>(a > (InEigenType)0 ? a : (-a)));
            };
            break;
        default:
            ERROR_IF(true, "unsupported TOSA_REF_TYPE %s", EnumNameTOSAREFTYPE(Dtype));
    }

    return 0;
}

template <int Rank, TOSA_REF_TYPE Dtype>
int OpBitwiseNot<Rank, Dtype>::register_fcn()
{
    switch (Dtype)
    {
        case TOSA_REF_TYPE_INT8:
        case TOSA_REF_TYPE_INT16:
        case TOSA_REF_TYPE_INT32:
            this->fcn = [](InEigenType a) -> OutEigenType { return ~a; };
            break;
        default:
            ERROR_IF(true, "unsupported TOSA_REF_TYPE %s", EnumNameTOSAREFTYPE(Dtype));
    }

    return 0;
}

template <int Rank, TOSA_REF_TYPE Dtype>
int OpCeil<Rank, Dtype>::register_fcn()
{
    switch (Dtype)
    {
        case TOSA_REF_TYPE_FP16:
        case TOSA_REF_TYPE_BF16:
        case TOSA_REF_TYPE_FP32:
            this->fcn = [](InEigenType a) -> OutEigenType {
                return static_cast<OutEigenType>(fpTrunc<Dtype>(ceil(a)));
            };
            break;
        case TOSA_REF_TYPE_FP64:
            this->fcn = [](InEigenType a) -> OutEigenType { return ceil(a); };
            break;
        default:
            ERROR_IF(true, "unsupported TOSA_REF_TYPE %s", EnumNameTOSAREFTYPE(Dtype));
    }

    return 0;
}

template <int Rank, TOSA_REF_TYPE Dtype>
int OpClz<Rank, Dtype>::register_fcn()
{
    int32_t num_bits;
    switch (Dtype)
    {
        case TOSA_REF_TYPE_INT32:
            num_bits = 32;
            break;
        default:
            ERROR_IF(true, "unsupported TOSA_REF_TYPE %s", EnumNameTOSAREFTYPE(Dtype));
    }

    this->fcn = [num_bits](int32_t a) -> int32_t {
        int32_t leading_zeros = 0;
        for (int bit = num_bits - 1; bit >= 0; bit--)
        {
            if (((a >> bit) & 0x1) == 0)
            {
                leading_zeros++;
            }
            else
            {
                break;
            }
        }
        return leading_zeros;
    };

    return 0;
}

template <int Rank, TOSA_REF_TYPE Dtype>
int OpCos<Rank, Dtype>::register_fcn()
{
    switch (Dtype)
    {
        case TOSA_REF_TYPE_FP16:
        case TOSA_REF_TYPE_BF16:
        case TOSA_REF_TYPE_FP32:
            this->fcn = [](InEigenType a) -> OutEigenType { return static_cast<OutEigenType>(fpTrunc<Dtype>(cos(a))); };
            break;
        case TOSA_REF_TYPE_FP64:
            if (g_func_config.bounds_mode)
            {
                // We don't need to calculate boundsValue for COS or SIN as they are both set as constants (1.0).
                // So instead we pass boundsMagnitude in place of boundsValue as the boundsMagnitude
                // depends on the input values with a different calculation that cannot be done at validation time.
                this->fcn = [](InEigenType a) -> OutEigenType {
                    return 1.0f + static_cast<OutEigenType>(a > static_cast<InEigenType>(0) ? a : (-a));
                };
            }
            else
            {
                this->fcn = [](InEigenType a) -> OutEigenType { return cos(a); };
            };
            break;
        default:
            ERROR_IF(true, "unsupported TOSA_REF_TYPE %s", EnumNameTOSAREFTYPE(Dtype));
    }

    return 0;
}

template <int Rank, TOSA_REF_TYPE Dtype>
int OpExp<Rank, Dtype>::register_fcn()
{
    switch (Dtype)
    {
        case TOSA_REF_TYPE_FP16:
        case TOSA_REF_TYPE_BF16:
        case TOSA_REF_TYPE_FP32:
            this->fcn = [](InEigenType a) -> OutEigenType { return static_cast<OutEigenType>(fpTrunc<Dtype>(exp(a))); };
            break;
        case TOSA_REF_TYPE_FP64:
            if (g_func_config.bounds_mode)
            {
                // ABS_ERROR bounds return (2*abs(a))
                // NOTE: err_base is added as part of the compliance tosa_verify
                this->fcn = [](InEigenType a) -> OutEigenType {
                    return static_cast<OutEigenType>(2.0) *
                           static_cast<OutEigenType>(a > static_cast<InEigenType>(0) ? a : (-a));
                };
            }
            else
            {
                this->fcn = [](InEigenType a) -> OutEigenType { return exp(a); };
            }
            break;
        default:
            ERROR_IF(true, "unsupported TOSA_REF_TYPE %s", EnumNameTOSAREFTYPE(Dtype));
    }

    return 0;
}

template <int Rank, TOSA_REF_TYPE Dtype>
int OpFloor<Rank, Dtype>::register_fcn()
{
    switch (Dtype)
    {
        case TOSA_REF_TYPE_FP16:
        case TOSA_REF_TYPE_BF16:
        case TOSA_REF_TYPE_FP32:
            this->fcn = [](InEigenType a) -> OutEigenType {
                return static_cast<OutEigenType>(fpTrunc<Dtype>(floor(a)));
            };
            break;
        case TOSA_REF_TYPE_FP64:
            this->fcn = [](InEigenType a) -> OutEigenType { return floor(a); };
            break;
        default:
            ERROR_IF(true, "unsupported TOSA_REF_TYPE %s", EnumNameTOSAREFTYPE(Dtype));
    }

    return 0;
}

template <int Rank, TOSA_REF_TYPE Dtype>
int OpLog<Rank, Dtype>::register_fcn()
{
    switch (Dtype)
    {
        case TOSA_REF_TYPE_FP16:
        case TOSA_REF_TYPE_BF16:
        case TOSA_REF_TYPE_FP32:
            this->fcn = [](InEigenType a) -> OutEigenType { return static_cast<OutEigenType>(fpTrunc<Dtype>(log(a))); };
            break;
        case TOSA_REF_TYPE_FP64:
            this->fcn = [](InEigenType a) -> OutEigenType { return log(a); };
            break;
        default:
            ERROR_IF(true, "unsupported TOSA_REF_TYPE %s", EnumNameTOSAREFTYPE(Dtype));
    }

    return 0;
}

template <int Rank, TOSA_REF_TYPE Dtype>
int OpLogicalNot<Rank, Dtype>::register_fcn()
{
    switch (Dtype)
    {
        case TOSA_REF_TYPE_BOOL:
            this->fcn = [](InEigenType a) -> OutEigenType { return !a; };
            break;
        default:
            ERROR_IF(true, "unsupported TOSA_REF_TYPE %s", EnumNameTOSAREFTYPE(Dtype));
    }

    return 0;
}

template <int Rank, TOSA_REF_TYPE Dtype>
OpNegate<Rank, Dtype>::OpNegate(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_)
    : UnaryNode<Rank, Dtype>(sgt_, Op_NEGATE, id_)
{
    this->setRequiredOperands(3, 1);

    register_fcn();
}

template <int Rank, TOSA_REF_TYPE Dtype>
OpNegate<Rank, Dtype>::~OpNegate()
{}

template <int Rank, TOSA_REF_TYPE Dtype>
int OpNegate<Rank, Dtype>::checkTensorAttributes()
{
    if (UnaryNode<Rank, Dtype>::checkTensorAttributes())
    {
        return 1;
    }

    if (this->validateRequiredRank(this->inputs[1], 1, 1) || this->validateRequiredRank(this->inputs[2], 1, 1))
    {
        this->printNodeValidationError("OpNegate: input and output zero point must be rank 1");
        return 1;
    }

    input_zp  = dynamic_cast<TosaReference::TensorTemplate<TInZp>*>(this->inputs[1]);
    output_zp = dynamic_cast<TosaReference::TensorTemplate<TOutZp>*>(this->inputs[2]);

    ASSERT_MEM(input_zp && output_zp);

    if (input_zp->getShape()[0] != 1)
    {
        this->printNodeValidationError("OpNegate: input zero point shape should be [1]");
        return 1;
    }

    if (output_zp->getShape()[0] != 1)
    {
        this->printNodeValidationError("OpNegate: output zero point shape should be [1]");
        return 1;
    }

    return 0;
}

template <int Rank, TOSA_REF_TYPE Dtype>
int OpNegate<Rank, Dtype>::register_fcn()
{
    switch (Dtype)
    {
        case TOSA_REF_TYPE_FP16:
        case TOSA_REF_TYPE_BF16:
        case TOSA_REF_TYPE_FP32:
            this->fcn = [this](InEigenType a) -> OutEigenType {
                const int64_t input_zp_val  = static_cast<int64_t>(this->input_zp->getTensor()(0));
                const int64_t output_zp_val = static_cast<int64_t>(this->output_zp->getTensor()(0));
                ERROR_IF(input_zp_val != 0, "OpNegate: Input zero point must be zero for non int8_t data");
                ERROR_IF(output_zp_val != 0, "OpNegate: Output zero point must be zero for non int8_t data");
                InEigenType result = -(a);
                return static_cast<OutEigenType>(fpTrunc<Dtype>(result));
            };
            break;
        case TOSA_REF_TYPE_FP64:
            this->fcn = [this](InEigenType a) -> OutEigenType {
                const int64_t input_zp_val  = static_cast<int64_t>(this->input_zp->getTensor()(0));
                const int64_t output_zp_val = static_cast<int64_t>(this->output_zp->getTensor()(0));
                ERROR_IF(input_zp_val != 0, "OpNegate: Input zero point must be zero for non int8_t data");
                ERROR_IF(output_zp_val != 0, "OpNegate: Output zero point must be zero for non int8_t data");
                OutEigenType result = -(a);
                return result;
            };
            break;
        case TOSA_REF_TYPE_INT16:
        case TOSA_REF_TYPE_INT32:
            this->fcn = [this](InEigenType a) -> OutEigenType {
                const int64_t input_zp_val  = static_cast<int64_t>(this->input_zp->getTensor()(0));
                const int64_t output_zp_val = static_cast<int64_t>(this->output_zp->getTensor()(0));
                ERROR_IF(input_zp_val != 0, "OpNegate: Input zero point must be zero for non int8_t data");
                ERROR_IF(output_zp_val != 0, "OpNegate: Output zero point must be zero for non int8_t data");
                int64_t res_in_64     = 0L - static_cast<int64_t>(a);
                int64_t i32_max_in_64 = static_cast<int64_t>(std::numeric_limits<int32_t>::max());
                int64_t i32_min_in_64 = static_cast<int64_t>(std::numeric_limits<int32_t>::min());
                REQUIRE(res_in_64 <= i32_max_in_64 && res_in_64 >= i32_min_in_64,
                        "OpNegate: result not in acc type range (int32)");

                int64_t max_clip_in_64, min_clip_in_64;
                if (Dtype == TOSA_REF_TYPE_INT16)
                {
                    max_clip_in_64 = static_cast<int64_t>(std::numeric_limits<int16_t>::max());
                    min_clip_in_64 = static_cast<int64_t>(std::numeric_limits<int16_t>::min());
                }
                else
                {
                    max_clip_in_64 = i32_max_in_64;
                    min_clip_in_64 = i32_min_in_64;
                }
                return static_cast<InEigenType>(
                    std::min<int64_t>(max_clip_in_64, std::max<int64_t>(min_clip_in_64, res_in_64)));
            };
            break;
        case TOSA_REF_TYPE_INT8:
            this->fcn = [this](InEigenType a) -> OutEigenType {
                const int64_t input_zp_val  = static_cast<int64_t>(this->input_zp->getTensor()(0));
                const int64_t output_zp_val = static_cast<int64_t>(this->output_zp->getTensor()(0));
                int64_t res_in_64           = 0 - (static_cast<int64_t>(a) - input_zp_val);
                int64_t i32_max_in_64       = static_cast<int64_t>(std::numeric_limits<int32_t>::max());
                int64_t i32_min_in_64       = static_cast<int64_t>(std::numeric_limits<int32_t>::min());
                REQUIRE(res_in_64 <= i32_max_in_64 && res_in_64 >= i32_min_in_64,
                        "OpNegate: result not in acc type range (int32)");
                res_in_64 += output_zp_val;
                InEigenType result = static_cast<InEigenType>(
                    std::min(std::max(res_in_64, static_cast<int64_t>(QMin)), static_cast<int64_t>(QMax)));
                return result;
            };
            break;
        default:
            ERROR_IF(true, "unsupported TOSA_REF_TYPE %s", EnumNameTOSAREFTYPE(Dtype));
    }

    return 0;
}

template <int Rank, TOSA_REF_TYPE Dtype>
int OpReciprocal<Rank, Dtype>::register_fcn()
{
    switch (Dtype)
    {
        case TOSA_REF_TYPE_FP16:
        case TOSA_REF_TYPE_BF16:
        case TOSA_REF_TYPE_FP32:
            this->fcn = [](InEigenType a) -> OutEigenType {
                return static_cast<OutEigenType>(fpTrunc<Dtype>(1.0 / a));
            };
            break;
        case TOSA_REF_TYPE_FP64:
            this->fcn = [](InEigenType a) -> OutEigenType { return static_cast<OutEigenType>(1.0L / a); };
            break;
        default:
            ERROR_IF(true, "unsupported TOSA_REF_TYPE %s", EnumNameTOSAREFTYPE(Dtype));
    }

    return 0;
}

template <int Rank, TOSA_REF_TYPE Dtype>
int OpRsqrt<Rank, Dtype>::register_fcn()
{
    switch (Dtype)
    {
        case TOSA_REF_TYPE_FP16:
        case TOSA_REF_TYPE_BF16:
        case TOSA_REF_TYPE_FP32:
            this->fcn = [](InEigenType a) -> OutEigenType {
                return static_cast<OutEigenType>(fpTrunc<Dtype>(1.0 / sqrt(a)));
            };
            break;
        case TOSA_REF_TYPE_FP64:
            this->fcn = [](InEigenType a) -> OutEigenType { return static_cast<OutEigenType>(1.0L / sqrt(a)); };
            break;
        default:
            ERROR_IF(true, "unsupported TOSA_REF_TYPE %s", EnumNameTOSAREFTYPE(Dtype));
    }

    return 0;
}

template <int Rank, TOSA_REF_TYPE Dtype>
int OpSin<Rank, Dtype>::register_fcn()
{
    switch (Dtype)
    {
        case TOSA_REF_TYPE_FP16:
        case TOSA_REF_TYPE_BF16:
        case TOSA_REF_TYPE_FP32:
            this->fcn = [](InEigenType a) -> OutEigenType { return static_cast<OutEigenType>(fpTrunc<Dtype>(sin(a))); };
            break;
        case TOSA_REF_TYPE_FP64:
            if (g_func_config.bounds_mode)
            {
                // We don't need to calculate boundsValue for COS or SIN as they are both set as constants (1.0).
                // So instead we pass boundsMagnitude in place of boundsValue as the boundsMagnitude
                // depends on the input values with a different calculation that cannot be done at validation time.
                this->fcn = [](InEigenType a) -> OutEigenType { return a; };
            }
            else
            {
                this->fcn = [](InEigenType a) -> OutEigenType { return sin(a); };
            };
            break;
        default:
            ERROR_IF(true, "unsupported TOSA_REF_TYPE %s", EnumNameTOSAREFTYPE(Dtype));
    }

    return 0;
}

// template explicit instantiation
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(UnaryNode, BOOL);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(UnaryNode, FP16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(UnaryNode, BF16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(UnaryNode, FP32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(UnaryNode, INT8);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(UnaryNode, INT16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(UnaryNode, INT32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(UnaryNode, FP64);

DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpAbs, FP16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpAbs, BF16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpAbs, FP32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpAbs, INT32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpAbs, FP64);

DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpBitwiseNot, INT8);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpBitwiseNot, INT16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpBitwiseNot, INT32);

DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpCeil, FP16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpCeil, BF16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpCeil, FP32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpCeil, FP64);

DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpClz, INT32);

DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpCos, FP16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpCos, BF16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpCos, FP32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpCos, FP64);

DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpExp, FP16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpExp, BF16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpExp, FP32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpExp, FP64);

DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpFloor, FP16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpFloor, BF16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpFloor, FP32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpFloor, FP64);

DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpLog, FP16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpLog, BF16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpLog, FP32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpLog, FP64);

DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpLogicalNot, BOOL);

DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpNegate, FP16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpNegate, BF16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpNegate, FP32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpNegate, INT8);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpNegate, INT16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpNegate, INT32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpNegate, FP64);

DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpRsqrt, FP16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpRsqrt, BF16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpRsqrt, FP32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpRsqrt, FP64);

DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpSin, FP16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpSin, BF16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpSin, FP32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpSin, FP64);

DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpReciprocal, FP16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpReciprocal, BF16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpReciprocal, FP32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpReciprocal, FP64);

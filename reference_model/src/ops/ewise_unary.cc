
// Copyright (c) 2020-2021, ARM Limited.
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

template <int Rank, DType Dtype>
UnaryNode<Rank, Dtype>::UnaryNode(SubgraphTraverser* sgt_, const Op& op_, uint64_t id_)
    : GraphNode(sgt_, op_, id_)
{
    setRequiredOperands(1, 1);
    setRequiredRank(0, 6);

    fcn = [](InEigenType a) -> OutEigenType {
        ASSERT_MSG(0, "In default UnaryNode function, missing function registration");
        return OutEigenType();
    };
}

template <int Rank, DType Dtype>
UnaryNode<Rank, Dtype>::~UnaryNode()
{}

template <int Rank, DType Dtype>
int UnaryNode<Rank, Dtype>::checkTensorAttributes()
{
    if (validateRequiredOperands())
        return 1;

    if (validateRequiredRank(inputs[0]) || validateRequiredRank(outputs[0]))
    {
        return 1;
    }

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

template <int Rank, DType Dtype>
int UnaryNode<Rank, Dtype>::eval()
{
    this->result->getTensor() = this->a->getTensor().unaryExpr(this->fcn);

    return GraphNode::eval();
}

template <int Rank, DType Dtype>
int OpAbs<Rank, Dtype>::register_fcn()
{
    switch (Dtype)
    {
        case DType_FLOAT:
        case DType_INT32:
            this->fcn = [](InEigenType a) -> OutEigenType { return a > (InEigenType)0 ? a : (-a); };
            break;
        default:
            ERROR_IF(true, "unsupported DType %s", EnumNamesDType()[Dtype]);
    }

    return 0;
}

template <int Rank, DType Dtype>
int OpBitwiseNot<Rank, Dtype>::register_fcn()
{
    switch (Dtype)
    {
        case DType_INT8:
        case DType_INT16:
        case DType_INT32:
            this->fcn = [](InEigenType a) -> OutEigenType { return ~a; };
            break;
        default:
            ERROR_IF(true, "unsupported DType %s", EnumNamesDType()[Dtype]);
    }

    return 0;
}

template <int Rank, DType Dtype>
int OpCeil<Rank, Dtype>::register_fcn()
{
    switch (Dtype)
    {
        case DType_FLOAT:
            this->fcn = [](InEigenType a) -> OutEigenType { return ceilf(a); };
            break;
        default:
            ERROR_IF(true, "unsupported DType %s", EnumNamesDType()[Dtype]);
    }

    return 0;
}

template <int Rank, DType Dtype>
int OpClz<Rank, Dtype>::register_fcn()
{
    int32_t num_bits;
    switch (Dtype)
    {
        case DType_INT32:
            num_bits = 32;
            break;
        default:
            ERROR_IF(true, "unsupported DType %s", EnumNamesDType()[Dtype]);
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

template <int Rank, DType Dtype>
int OpExp<Rank, Dtype>::register_fcn()
{
    switch (Dtype)
    {
        case DType_FLOAT:
            this->fcn = [](InEigenType a) -> OutEigenType { return expf(a); };
            break;
        default:
            ERROR_IF(true, "unsupported DType %s", EnumNamesDType()[Dtype]);
    }

    return 0;
}

template <int Rank, DType Dtype>
int OpFloor<Rank, Dtype>::register_fcn()
{
    switch (Dtype)
    {
        case DType_FLOAT:
            this->fcn = [](InEigenType a) -> OutEigenType { return floorf(a); };
            break;
        default:
            ERROR_IF(true, "unsupported DType %s", EnumNamesDType()[Dtype]);
    }

    return 0;
}

template <int Rank, DType Dtype>
int OpLog<Rank, Dtype>::register_fcn()
{
    switch (Dtype)
    {
        case DType_FLOAT:
            this->fcn = [](InEigenType a) -> OutEigenType { return logf(a); };
            break;
        default:
            ERROR_IF(true, "unsupported DType %s", EnumNamesDType()[Dtype]);
    }

    return 0;
}

template <int Rank, DType Dtype>
int OpLogicalNot<Rank, Dtype>::register_fcn()
{
    switch (Dtype)
    {
        case DType_BOOL:
            this->fcn = [](InEigenType a) -> OutEigenType { return !a; };
            break;
        default:
            ERROR_IF(true, "unsupported DType %s", EnumNamesDType()[Dtype]);
    }

    return 0;
}

template <int Rank, DType Dtype>
OpNegate<Rank, Dtype>::OpNegate(SubgraphTraverser* sgt_,
                                TosaAttributeBase* attribute_,
                                uint64_t id_)
    : UnaryNode<Rank, Dtype>(sgt_, Op_NEGATE, id_)
{
    INIT_ATTRIBUTE(Negate);

    register_fcn();
}

template <int Rank, DType Dtype>
OpNegate<Rank, Dtype>::~OpNegate()
{
    if (attribute)
        delete attribute;
}

template <int Rank, DType Dtype>
int OpNegate<Rank, Dtype>::register_fcn()
{
    ERROR_IF(Dtype != DType_INT8 && attribute->input1_zp() != 0, "OpNegate: zeropoint only for int8_t");
    ERROR_IF(Dtype != DType_INT8 && attribute->output_zp() != 0, "OpNegate: zeropoint only for int8_t");

    switch (Dtype)
    {
        case DType_FLOAT:
            this->fcn = [](InEigenType a) -> OutEigenType {
                InEigenType result = -(a);
                return result;
            };
            break;
        case DType_INT16:
        case DType_INT32:
            this->fcn = [this](InEigenType a) -> OutEigenType {
                int64_t res_in_64 = 0L - a;
                int64_t i32_max_in_64 = static_cast<int64_t>(std::numeric_limits<int32_t>::max());
                int64_t i32_min_in_64 = static_cast<int64_t>(std::numeric_limits<int32_t>::min());
                REQUIRE(res_in_64 <= i32_max_in_64 && res_in_64 >= i32_min_in_64, "OpNegate: result not in acc type range (int32)");

                int64_t max_clip_in_64, min_clip_in_64;
                if (Dtype == DType_INT16)
                {
                    max_clip_in_64 = static_cast<int64_t>(std::numeric_limits<int16_t>::max());
                    min_clip_in_64 = static_cast<int64_t>(std::numeric_limits<int16_t>::min());
                }
                else
                {
                    max_clip_in_64 = i32_max_in_64;
                    min_clip_in_64 = i32_min_in_64;
                }
                return static_cast<InEigenType>(std::min<int64_t>(max_clip_in_64, std::max<int64_t>(min_clip_in_64, res_in_64)));
            };
            break;
        case DType_INT8:
            this->fcn = [this](InEigenType a) -> OutEigenType {
                int64_t res_in_64 = 0 - (a - attribute->input1_zp());
                int64_t i32_max_in_64 = static_cast<int64_t>(std::numeric_limits<int32_t>::max());
                int64_t i32_min_in_64 = static_cast<int64_t>(std::numeric_limits<int32_t>::min());
                REQUIRE(res_in_64 <= i32_max_in_64 && res_in_64 >= i32_min_in_64, "OpNegate: result not in acc type range (int32)");
                res_in_64 += attribute->output_zp();
                InEigenType result = static_cast<InEigenType>(std::min(std::max(res_in_64, static_cast<int64_t>(QMin)), static_cast<int64_t>(QMax)));
                return result;
            };
            break;
        default:
            ERROR_IF(true, "unsupported DType %s", EnumNamesDType()[Dtype]);
    }

    return 0;
}

template <int Rank, DType Dtype>
int OpReciprocal<Rank, Dtype>::register_fcn()
{
    switch (Dtype)
    {
        case DType_FLOAT:
            this->fcn = [](InEigenType a) -> OutEigenType { return 1.0 / a; };
            break;
        default:
            ERROR_IF(true, "unsupported DType %s", EnumNamesDType()[Dtype]);
    }

    return 0;
}

template <int Rank, DType Dtype>
int OpRsqrt<Rank, Dtype>::register_fcn()
{
    switch (Dtype)
    {
        case DType_FLOAT:
            this->fcn = [](InEigenType a) -> OutEigenType { return 1.0 / sqrtf(a); };
            break;
        default:
            ERROR_IF(true, "unsupported DType %s", EnumNamesDType()[Dtype]);
    }

    return 0;
}

// template explicit instantiation
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpAbs, FLOAT);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpAbs, INT32);

DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpBitwiseNot, INT8);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpBitwiseNot, INT16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpBitwiseNot, INT32);

DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpCeil, FLOAT);

DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpClz, INT32);

DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpExp, FLOAT);

DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpFloor, FLOAT);

DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpLog, FLOAT);

DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpLogicalNot, BOOL);

DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpNegate, FLOAT);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpNegate, INT8);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpNegate, INT16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpNegate, INT32);

DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpRsqrt, FLOAT);

DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpReciprocal, FLOAT);

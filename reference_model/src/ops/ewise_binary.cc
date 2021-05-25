
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

#include "ewise_binary.h"
#include "arith_util.h"
#include "quant_util.h"
#include "template_types.h"

using namespace TosaReference;
using namespace Eigen;
using namespace tosa;

template <int Rank, DType InDtype, DType OutDtype>
BinaryNodeBase<Rank, InDtype, OutDtype>::BinaryNodeBase(const Op& op_, TosaQuantInfoBase* qinfo_, uint64_t id_)
    : GraphNode(op_, id_)
{
    setRequiredOperands(2, 1);
    setRequiredRank(0, 6);

    a_rank = b_rank = max_input_rank = -1;
    a = b   = nullptr;
    a_rank0 = b_rank0 = nullptr;
    result            = nullptr;

    fcn = [](InEigenType a, InEigenType b) -> OutEigenType { return OutEigenType(); };
}

template <int Rank, DType InDtype, DType OutDtype>
BinaryNodeBase<Rank, InDtype, OutDtype>::~BinaryNodeBase()
{}

template <int Rank, DType InDtype, DType OutDtype>
int BinaryNodeBase<Rank, InDtype, OutDtype>::checkTensorAttributes()
{
    if (validateRequiredOperands())
        return 1;

    if (validateRequiredRank(inputs[0]) || validateRequiredRank(inputs[1]) || validateRequiredRank(outputs[0]))
    {
        return 1;
    }

    a_rank = inputs[0]->getRank();
    b_rank = inputs[1]->getRank();
    if (a_rank != 0 && b_rank != 0 && a_rank != b_rank)
    {
        printNodeValidationError("Binary operator input ranks must match");
        return 1;
    }

    max_input_rank = a_rank >= b_rank ? a_rank : b_rank;

    // A & B must be the same types
    if (inputs[0]->matchType(*inputs[1]))
    {
        printNodeValidationError("Binary operator input types must match");
        return 1;
    }

    // Result's geometry must match, but the type may be wider
    if (outputs[0]->getRank() != max_input_rank)
    {
        printNodeValidationError("Binary operator input and output genometry must match");
        return 1;
    }

    if (a_rank == max_input_rank)
    {
        a = dynamic_cast<TosaReference::TensorTemplate<TIn>*>(inputs[0]);
    }
    else
    {
        a_rank0 = dynamic_cast<TosaReference::TensorTemplate<ETensor0<InEigenType>>*>(inputs[0]);
    }

    if (b_rank == max_input_rank)
    {
        b = dynamic_cast<TosaReference::TensorTemplate<TIn>*>(inputs[1]);
    }
    else
    {
        b_rank0 = dynamic_cast<TosaReference::TensorTemplate<ETensor0<InEigenType>>*>(inputs[1]);
    }

    result = dynamic_cast<TosaReference::TensorTemplate<TOut>*>(outputs[0]);

    // either a or b can be rank0
    // a_rank0 and b_rank0 can't be valid at the same time.
    // if a and be are both rank0, they should be evaulated as 'a' and 'b', instead of 'a_rank0' and 'b_rank0'
    ASSERT_MEM((a || a_rank0) && (b || b_rank0) && !(a_rank0 && b_rank0) && result);

    return 0;
}

template <int Rank, DType InDtype, DType OutDtype>
int BinaryNodeBase<Rank, InDtype, OutDtype>::broadcast()
{
    auto output_shape = result->getTensor().dimensions();

    std::vector<int> a_shape, b_shape;

    if (a_rank == max_input_rank)
    {
        a_shape = a->getShape();
    }
    else
    {
        a_shape.assign(max_input_rank, 1);
    }

    if (b_rank == max_input_rank)
    {
        b_shape = b->getShape();
    }
    else
    {
        b_shape.assign(max_input_rank, 1);
    }

    for (int i = 0; i < max_input_rank; i++)
    {
        if (a_shape[i] != output_shape[i] && a_shape[i] == 1)
        {
            bcast_a[i] = output_shape[i];
        }
        else
        {
            bcast_a[i] = 1;
        }
        if (b_shape[i] != output_shape[i] && b_shape[i] == 1)
        {
            bcast_b[i] = output_shape[i];
        }
        else
        {
            bcast_b[i] = 1;
        }
    }

    return 0;
}

template <int Rank, DType InDtype, DType OutDtype>
int BinaryNode<Rank, InDtype, OutDtype>::eval()
{
    this->broadcast();

    Eigen::array<int, Rank> reshaper;
    reshaper.fill(1);
    TIn ia, ib;

    if (this->a_rank == this->max_input_rank)
    {
        ia = this->a->getTensor().broadcast(this->bcast_a);
    }
    else
    {
        ia = this->a_rank0->getTensor().reshape(reshaper).broadcast(this->bcast_a);
    }

    if (this->b_rank == this->max_input_rank)
    {
        ib = this->b->getTensor().broadcast(this->bcast_b);
    }
    else
    {
        ib = this->b_rank0->getTensor().reshape(reshaper).broadcast(this->bcast_b);
    }

    this->result->getTensor() = ia.binaryExpr(ib, this->fcn);

    return GraphNode::eval();
}

// still need to partial specialize this, or Eigen will throw static assertion
template <DType InDtype, DType OutDtype>
int BinaryNode<0, InDtype, OutDtype>::eval()
{
    this->result->getTensor() = this->a->getTensor().binaryExpr(this->b->getTensor(), this->fcn);

    return GraphNode::eval();
}

template <int Rank, DType Dtype>
int OpAdd<Rank, Dtype>::register_fcn()
{
    switch (InDtype)
    {
        case DType_FLOAT:
        case DType_INT32:
            this->fcn = [](InEigenType a, InEigenType b) -> OutEigenType { return a + b; };
            break;
        default:
            FATAL_ERROR_NODE("unsupported DType %s", EnumNamesDType()[InDtype]);
    }

    return 0;
}

template <int Rank, DType Dtype>
int OpArithmeticRightShift<Rank, Dtype>::register_fcn()
{
    bool round       = attribute->round();
    int32_t num_bits = 0;
    switch (Dtype)
    {
        case DType_INT8:
            num_bits = 8;
            break;
        case DType_INT16:
            num_bits = 16;
            break;
        case DType_INT32:
            num_bits = 32;
            break;
        default:
            FATAL_ERROR_NODE("unsupported DType %s", EnumNamesDType()[Dtype]);
    }

    this->fcn = [this, round, num_bits](InEigenType a, InEigenType b) -> OutEigenType {
        ASSERT_MSG_NODE(b >= 0 && b < num_bits, "OpArithmeticRightShift: shift value %d is out of valid range [0, %d]",
                        (int32_t)b, num_bits);

        InEigenType acc = a >> b;

        if (round && b > 0 && (a >> (b - 1) & 1) != 0)
        {
            acc++;
        }

        return acc;
    };

    return 0;
}

template <int Rank, DType Dtype>
int OpBitwiseAnd<Rank, Dtype>::register_fcn()
{
    switch (Dtype)
    {
        case DType_INT8:
        case DType_INT16:
        case DType_INT32:
            this->fcn = [](InEigenType a, InEigenType b) -> OutEigenType { return a & b; };
            break;
        default:
            FATAL_ERROR_NODE("unsupported DType %s", EnumNamesDType()[Dtype]);
    }

    return 0;
}

template <int Rank, DType Dtype>
int OpBitwiseOr<Rank, Dtype>::register_fcn()
{
    switch (Dtype)
    {
        case DType_INT8:
        case DType_INT16:
        case DType_INT32:
            this->fcn = [](InEigenType a, InEigenType b) -> OutEigenType { return a | b; };
            break;
        default:
            FATAL_ERROR_NODE("unsupported DType %s", EnumNamesDType()[Dtype]);
    }

    return 0;
}

template <int Rank, DType Dtype>
int OpBitwiseXor<Rank, Dtype>::register_fcn()
{
    switch (Dtype)
    {
        case DType_INT8:
        case DType_INT16:
        case DType_INT32:
            this->fcn = [](InEigenType a, InEigenType b) -> OutEigenType { return a ^ b; };
            break;
        default:
            FATAL_ERROR_NODE("unsupported DType %s", EnumNamesDType()[Dtype]);
    }

    return 0;
}

template <int Rank, DType Dtype>
int OpDiv<Rank, Dtype>::register_fcn()
{
    switch (InDtype)
    {
        case DType_INT32:
            this->fcn = [this](InEigenType a, InEigenType b) -> OutEigenType {
                ASSERT_MSG_NODE(b != 0, "OpDiv: divisor must be non-zero value");
                int64_t res_in_64     = static_cast<int64_t>(a) / b;
                int64_t i32_max_in_64 = static_cast<int64_t>(std::numeric_limits<InEigenType>::max());
                ASSERT_MSG_NODE(a <= i32_max_in_64, "OpDiv: result not in i32 range");
                return static_cast<InEigenType>(res_in_64);
            };
            break;
        default:
            FATAL_ERROR_NODE("unsupported DType %s", EnumNamesDType()[InDtype]);
    }

    return 0;
}

template <int Rank, DType Dtype>
int OpLogicalAnd<Rank, Dtype>::register_fcn()
{
    switch (Dtype)
    {
        case DType_BOOL:
            this->fcn = [](InEigenType a, InEigenType b) -> OutEigenType { return a && b; };
            break;
        default:
            FATAL_ERROR_NODE("unsupported DType %s", EnumNamesDType()[Dtype]);
    }

    return 0;
}

template <int Rank, DType Dtype>
int OpLogicalLeftShift<Rank, Dtype>::register_fcn()
{
    switch (Dtype)
    {
        case DType_INT8:
        case DType_INT16:
        case DType_INT32:
            this->fcn = [](InEigenType a, InEigenType b) -> OutEigenType { return a << b; };
            break;
        default:
            FATAL_ERROR_NODE("unsupported DType %s", EnumNamesDType()[Dtype]);
    }

    return 0;
}

template <int Rank, DType Dtype>
int OpLogicalRightShift<Rank, Dtype>::register_fcn()
{
    int32_t num_bits = 0;
    switch (Dtype)
    {
        case DType_INT8:
            num_bits = 8;
            break;
        case DType_INT16:
            num_bits = 16;
            break;
        case DType_INT32:
            num_bits = 32;
            break;
        default:
            FATAL_ERROR_NODE("unsupported DType %s", EnumNamesDType()[Dtype]);
    }

    this->fcn = [num_bits](InEigenType a, InEigenType b) -> OutEigenType {
        uint32_t mask = ONES_MASK(num_bits) >> b;
        return (a >> b) & mask;
    };

    return 0;
}

template <int Rank, DType Dtype>
int OpLogicalOr<Rank, Dtype>::register_fcn()
{
    switch (Dtype)
    {
        case DType_BOOL:
            this->fcn = [](InEigenType a, InEigenType b) -> OutEigenType { return a || b; };
            break;
        default:
            FATAL_ERROR_NODE("unsupported DType %s", EnumNamesDType()[Dtype]);
    }

    return 0;
}

template <int Rank, DType Dtype>
int OpLogicalXor<Rank, Dtype>::register_fcn()
{
    switch (Dtype)
    {
        case DType_BOOL:
            this->fcn = [](InEigenType a, InEigenType b) -> OutEigenType { return a ^ b; };
            break;
        default:
            FATAL_ERROR_NODE("unsupported DType %s", EnumNamesDType()[Dtype]);
    }

    return 0;
}

template <int Rank, DType Dtype>
int OpMaximum<Rank, Dtype>::register_fcn()
{
    switch (Dtype)
    {
        case DType_FLOAT:
        case DType_INT32:
            this->fcn = [](InEigenType a, InEigenType b) -> OutEigenType { return a > b ? a : b; };
            break;
        default:
            FATAL_ERROR_NODE("unsupported DType %s", EnumNamesDType()[Dtype]);
    }

    return 0;
}

template <int Rank, DType Dtype>
int OpMinimum<Rank, Dtype>::register_fcn()
{
    switch (Dtype)
    {
        case DType_FLOAT:
        case DType_INT32:
            this->fcn = [](InEigenType a, InEigenType b) -> OutEigenType { return a < b ? a : b; };
            break;
        default:
            FATAL_ERROR_NODE("unsupported DType %s", EnumNamesDType()[Dtype]);
    }

    return 0;
}

template <int Rank, DType InDtype, DType OutDtype>
int OpMul<Rank, InDtype, OutDtype>::register_fcn()
{
    int32_t shift = attribute->shift();
    ASSERT_MSG_NODE(InDtype == DType_INT32 || shift == 0, "OpMul: shift needs to be 0 but is %d if input is %s", shift,
                    EnumNamesDType()[InDtype]);

    switch (InDtype)
    {
        case DType_FLOAT:
            this->fcn = [shift](InEigenType a, InEigenType b) -> OutEigenType { return a * b; };
            break;
        case DType_INT32:
            this->fcn = [this, shift](InEigenType a, InEigenType b) -> OutEigenType {
                int64_t result;
                if (shift > 0)
                {
                    int64_t round = 1L << (shift - 1);
                    result        = static_cast<int64_t>(a) * static_cast<int64_t>(b) + round;
                    result        = result >> shift;

                    ASSERT_MSG_NODE(result >= QMin && result <= QMax,
                                    "OpMul: result %ld exceeds valid range [%ld, %ld]", result, QMin, QMax);
                }
                else
                {
                    result = a * b;
                }

                return static_cast<OutEigenType>(result);
            };
            break;
        case DType_INT8:
        case DType_INT16:
            this->fcn = [this](InEigenType lhs, InEigenType rhs) -> OutEigenType {
                OutEigenType raw_output = (OutEigenType)lhs * (OutEigenType)rhs;

                OutEigenType clamped_output = std::min<OutEigenType>(QMax, std::max<OutEigenType>(raw_output, QMin));

                return clamped_output;
            };
            break;
        default:
            FATAL_ERROR_NODE("unsupported DType %s", EnumNamesDType()[InDtype]);
    }

    return 0;
}

template <int Rank, DType Dtype>
int OpPow<Rank, Dtype>::register_fcn()
{
    switch (Dtype)
    {
        case DType_FLOAT:
            this->fcn = [](InEigenType a, InEigenType b) -> OutEigenType { return powf(a, b); };
            break;
        default:
            FATAL_ERROR_NODE("unsupported DType %s", EnumNamesDType()[Dtype]);
    }

    return 0;
}

template <int Rank, DType Dtype>
int OpSub<Rank, Dtype>::register_fcn()
{
    switch (InDtype)
    {
        case DType_FLOAT:
        case DType_INT32:
            this->fcn = [](InEigenType a, InEigenType b) -> OutEigenType { return a - b; };
            break;
        default:
            FATAL_ERROR_NODE("unsupported DType %s", EnumNamesDType()[InDtype]);
    }

    return 0;
}

template <int Rank, DType InDtype>
OpTable<Rank, InDtype>::OpTable(TosaAttributeBase* attribute_, TosaQuantInfoBase* qinfo_, uint64_t id_)
    : GraphNode(Op_TABLE, id_)
{
    setRequiredOperands(2, 1);
    setRequiredRank(0, 6);
}

template <int Rank, DType InDtype>
OpTable<Rank, InDtype>::~OpTable()
{}

template <int Rank, DType InDtype>
int OpTable<Rank, InDtype>::checkTensorAttributes()
{
    if (validateRequiredOperands())
        return 1;

    if (validateRequiredRank(inputs[0]) || validateRequiredRank(outputs[0]))
    {
        return 1;
    }

    if (inputs[1]->getRank() != 1)
    {
        printNodeValidationError("OpTable: Table must be rank 1 tensor");
        return 1;
    }

    if (inputs[0]->getDtype() == DType_INT8)
    {
        if (inputs[1]->getElementCount() != 256 || inputs[1]->getDtype() != DType_INT8)
        {
            printNodeValidationError("OpTable: Table must be INT8[256] if input is INT8");
            return 1;
        }
    }
    else if (inputs[0]->getDtype() == DType_INT16)
    {
        if (inputs[1]->getElementCount() != 513 || inputs[1]->getDtype() != DType_INT16)
        {
            printNodeValidationError("OpTable: Table must be INT16[513] if input is INT16");
            return 1;
        }
    }

    in    = dynamic_cast<TosaReference::TensorTemplate<TIn>*>(inputs[0]);
    table = dynamic_cast<TosaReference::TensorTemplate<TTable>*>(inputs[1]);
    out   = dynamic_cast<TosaReference::TensorTemplate<TOut>*>(outputs[0]);

    ASSERT_MEM(in && table && out);

    return 0;
}

template <int Rank, DType InDtype>
int OpTable<Rank, InDtype>::eval()
{
    switch (InDtype)
    {
        case DType_INT8:
            this->out->getTensor() = this->in->getTensor().unaryExpr([this](InEigenType in) -> OutEigenType {
                int32_t input_truncated = std::min<int32_t>(std::max<int32_t>(in, QInMin), QInMax);
                int32_t index           = input_truncated - QInMin;
                int32_t value           = this->table->getTensor()(index);

                return value;
            });
            break;
        case DType_INT16:
            this->out->getTensor() = this->in->getTensor().unaryExpr([this](InEigenType in) -> OutEigenType {
                // 1. make sure input is int16 range
                int32_t input_truncated = std::min<int32_t>(std::max<int32_t>(in, QInMin), QInMax);

                // 2. calculate index and interpolation fraction
                int32_t index = (input_truncated >> FractionBits) + (1 << (IntegerBits - 1));
                index         = std::min<int32_t>(std::max<int32_t>(index, 0), NumTableEntries - 1);    // 9-bit index
                int32_t frac  = (input_truncated)&0x7F;    // 7-bit fraction

                // 3. interpolate, generate 16.7 (23-bit) output
                int32_t base  = this->table->getTensor()(index);
                int32_t next  = this->table->getTensor()(index + 1);
                int32_t value = (base << 7) + (next - base) * frac;

                return value;
            });
            break;
        default:
            FATAL_ERROR_NODE("unsupported DType %s", EnumNamesDType()[InDtype]);
    }

    return GraphNode::eval();
}

// template explicit instantiation
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpAdd, FLOAT);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpAdd, INT32);

DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpArithmeticRightShift, INT8);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpArithmeticRightShift, INT16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpArithmeticRightShift, INT32);

DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpBitwiseAnd, INT8);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpBitwiseAnd, INT16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpBitwiseAnd, INT32);

DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpBitwiseOr, INT8);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpBitwiseOr, INT16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpBitwiseOr, INT32);

DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpBitwiseXor, INT8);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpBitwiseXor, INT16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpBitwiseXor, INT32);

DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpDiv, INT32);

DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpLogicalAnd, BOOL);

DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpLogicalLeftShift, INT8);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpLogicalLeftShift, INT16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpLogicalLeftShift, INT32);

DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpLogicalRightShift, INT8);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpLogicalRightShift, INT16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpLogicalRightShift, INT32);

DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpLogicalOr, BOOL);

DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpLogicalXor, BOOL);

DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpMaximum, FLOAT);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpMaximum, INT32);

DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpMinimum, FLOAT);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpMinimum, INT32);

DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpMul, FLOAT, FLOAT);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpMul, INT8, INT32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpMul, INT16, INT32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpMul, INT32, INT32);

DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpPow, FLOAT);

DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpSub, FLOAT);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpSub, INT32);

DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpTable, INT8);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpTable, INT16);

DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(BinaryNode, FLOAT, BOOL);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(BinaryNode, INT32, BOOL);

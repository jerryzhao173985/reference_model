
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

#include "ewise_binary.h"
#include "arith_util.h"
#include "quant_util.h"
#include "template_types.h"

using namespace TosaReference;
using namespace Eigen;
using namespace tosa;

template <int Rank, TOSA_REF_TYPE InDtype, TOSA_REF_TYPE OutDtype>
BinaryNodeBase<Rank, InDtype, OutDtype>::BinaryNodeBase(SubgraphTraverser* sgt_, const Op& op_, uint64_t id_)
    : GraphNode(sgt_, op_, id_)
{
    setRequiredOperands(2, 1);

    a = b  = nullptr;
    result = nullptr;

    fcn = [](InEigenType a, InEigenType b) -> OutEigenType { return OutEigenType(); };
}

template <int Rank, TOSA_REF_TYPE InDtype, TOSA_REF_TYPE OutDtype>
BinaryNodeBase<Rank, InDtype, OutDtype>::~BinaryNodeBase()
{}

template <int Rank, TOSA_REF_TYPE InDtype, TOSA_REF_TYPE OutDtype>
int BinaryNodeBase<Rank, InDtype, OutDtype>::checkTensorAttributes()
{
    // Check Tosa Level
    auto tosa_level = g_func_config.tosa_level;
    LEVEL_CHECK(Rank <= tosa_level.MAX_RANK, "Rank should be smaller than or equal to MAX_RANK");

    if (validateRequiredOperands())
        return 1;

    // A & B must be the same rank and types
    if (inputs[0]->matchRankType(*inputs[1]))
    {
        printNodeValidationError("Binary operator input types must match");
        return 1;
    }

    if (inputs[0]->matchRankShape(*outputs[0], true /* broadcastOk */))
    {
        std::string err =
            "Binary operators " + std::string(EnumNamesOp()[nodeType]) + " lhs input and output rank/shape must match";
        printNodeValidationError(err.c_str());
        return 1;
    }

    if (inputs[1]->matchRankShape(*outputs[0], true /* broadcastOk */))
    {
        std::string err =
            "Binary operators " + std::string(EnumNamesOp()[nodeType]) + " rhs input and output rank/shape must match";
        printNodeValidationError(err.c_str());
        return 1;
    }

    ERROR_IF(outputs[0]->getDtype() != OutDtype, "Binary operator type doesn't match");

    a      = dynamic_cast<TosaReference::TensorTemplate<TIn>*>(inputs[0]);
    b      = dynamic_cast<TosaReference::TensorTemplate<TIn>*>(inputs[1]);
    result = dynamic_cast<TosaReference::TensorTemplate<TOut>*>(outputs[0]);

    ASSERT_MEM(a && b && result);

    return 0;
}

template <int Rank, TOSA_REF_TYPE InDtype, TOSA_REF_TYPE OutDtype>
int BinaryNodeBase<Rank, InDtype, OutDtype>::broadcast(std::vector<int>& calculated_shape)
{
    const std::vector<int>& a_shape      = a->getShape();
    const std::vector<int>& b_shape      = b->getShape();
    const std::vector<int>& output_shape = result->getShape();

    // calculates the multipliers for Eigen
    for (int i = 0; i < Rank; i++)
    {
        bcast_a[i] = (a_shape[i] != output_shape[i] && a_shape[i] == 1) ? output_shape[i] : 1;
        bcast_b[i] = (b_shape[i] != output_shape[i] && b_shape[i] == 1) ? output_shape[i] : 1;
    }

    // calculates the broadcasted output shape
    calculated_shape = a_shape;
    for (size_t i = 0; i < calculated_shape.size(); i++)
    {
        if (calculated_shape[i] == 1)
        {
            calculated_shape[i] = b_shape[i];
        }
        else
        {
            ERROR_IF(b_shape[i] != 1 && b_shape[i] != calculated_shape[i],
                     "Broadcast_shape failure, input shapes are not compatible");
        }
    }

    return 0;
}

template <int Rank, TOSA_REF_TYPE InDtype, TOSA_REF_TYPE OutDtype>
int BinaryNode<Rank, InDtype, OutDtype>::eval()
{
    std::vector<int> calculated_shape;
    this->broadcast(calculated_shape);

    auto result_shape = this->result->getShape();
    ERROR_IF(calculated_shape != result_shape,
             "Broadcast_shape failure, calculated_shape and result_shape don't match");

    Eigen::array<int, Rank> reshaper;
    reshaper.fill(1);
    TIn ia, ib;

    ia = this->a->getTensor().broadcast(this->bcast_a);
    ib = this->b->getTensor().broadcast(this->bcast_b);

    this->result->getTensor() = ia.binaryExpr(ib, this->fcn);

    return GraphNode::eval();
}

// still need to partial specialize this, or Eigen will throw static assertion
template <TOSA_REF_TYPE InDtype, TOSA_REF_TYPE OutDtype>
int BinaryNode<0, InDtype, OutDtype>::eval()
{
    this->result->getTensor() = this->a->getTensor().binaryExpr(this->b->getTensor(), this->fcn);

    return GraphNode::eval();
}

template <int Rank, TOSA_REF_TYPE Dtype>
int OpAdd<Rank, Dtype>::register_fcn()
{
    switch (InDtype)
    {
        case TOSA_REF_TYPE_INT32:
            this->fcn = [this](InEigenType a, InEigenType b) -> OutEigenType {
                int64_t res_in_64     = static_cast<int64_t>(a) + b;
                int64_t i32_max_in_64 = static_cast<int64_t>(std::numeric_limits<InEigenType>::max());
                int64_t i32_min_in_64 = static_cast<int64_t>(std::numeric_limits<InEigenType>::min());
                REQUIRE(res_in_64 <= i32_max_in_64 && res_in_64 >= i32_min_in_64, "OpAdd: result not in i32 range");
                return static_cast<InEigenType>(res_in_64);
            };
            break;
        case TOSA_REF_TYPE_FP16:
        case TOSA_REF_TYPE_BF16:
        case TOSA_REF_TYPE_FP32:
            this->fcn = [](InEigenType a, InEigenType b) -> OutEigenType { return fpTrunc<OutDtype>(a + b); };
            break;
        case TOSA_REF_TYPE_FP64:
            this->fcn = [](InEigenType a, InEigenType b) -> OutEigenType { return a + b; };
            break;
        default:
            ERROR_IF(true, "unsupported TOSA_REF_TYPE %s", EnumNameTOSAREFTYPE(InDtype));
    }

    return 0;
}

template <int Rank, TOSA_REF_TYPE Dtype>
int OpArithmeticRightShift<Rank, Dtype>::register_fcn()
{
    bool round       = attribute->round();
    int32_t num_bits = 0;
    switch (Dtype)
    {
        case TOSA_REF_TYPE_INT8:
            num_bits = 8;
            break;
        case TOSA_REF_TYPE_INT16:
            num_bits = 16;
            break;
        case TOSA_REF_TYPE_INT32:
            num_bits = 32;
            break;
        default:
            ERROR_IF(true, "unsupported TOSA_REF_TYPE %s", EnumNameTOSAREFTYPE(Dtype));
    }

    this->fcn = [this, round, num_bits](InEigenType a, InEigenType b) -> OutEigenType {
        REQUIRE(b >= 0 && b < num_bits, "OpArithmeticRightShift: shift value %d is out of valid range [0, %d]",
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

template <int Rank, TOSA_REF_TYPE Dtype>
OpArithmeticRightShift<Rank, Dtype>::~OpArithmeticRightShift()
{
    if (attribute)
        delete attribute;
}

template <int Rank, TOSA_REF_TYPE Dtype>
int OpBitwiseAnd<Rank, Dtype>::register_fcn()
{
    switch (Dtype)
    {
        case TOSA_REF_TYPE_INT8:
        case TOSA_REF_TYPE_INT16:
        case TOSA_REF_TYPE_INT32:
            this->fcn = [](InEigenType a, InEigenType b) -> OutEigenType { return a & b; };
            break;
        default:
            ERROR_IF(true, "unsupported TOSA_REF_TYPE %s", EnumNameTOSAREFTYPE(Dtype));
    }

    return 0;
}

template <int Rank, TOSA_REF_TYPE Dtype>
int OpBitwiseOr<Rank, Dtype>::register_fcn()
{
    switch (Dtype)
    {
        case TOSA_REF_TYPE_INT8:
        case TOSA_REF_TYPE_INT16:
        case TOSA_REF_TYPE_INT32:
            this->fcn = [](InEigenType a, InEigenType b) -> OutEigenType { return a | b; };
            break;
        default:
            ERROR_IF(true, "unsupported TOSA_REF_TYPE %s", EnumNameTOSAREFTYPE(Dtype));
    }

    return 0;
}

template <int Rank, TOSA_REF_TYPE Dtype>
int OpBitwiseXor<Rank, Dtype>::register_fcn()
{
    switch (Dtype)
    {
        case TOSA_REF_TYPE_INT8:
        case TOSA_REF_TYPE_INT16:
        case TOSA_REF_TYPE_INT32:
            this->fcn = [](InEigenType a, InEigenType b) -> OutEigenType { return a ^ b; };
            break;
        default:
            ERROR_IF(true, "unsupported TOSA_REF_TYPE %s", EnumNameTOSAREFTYPE(Dtype));
    }

    return 0;
}

template <int Rank, TOSA_REF_TYPE Dtype>
int OpIntdiv<Rank, Dtype>::register_fcn()
{
    switch (InDtype)
    {
        case TOSA_REF_TYPE_INT32:
            this->fcn = [this](InEigenType a, InEigenType b) -> OutEigenType {
                REQUIRE(b != 0, "OpIntDiv: divisor must be non-zero value");
                int64_t res_in_64     = static_cast<int64_t>(a) / b;
                int64_t i32_max_in_64 = static_cast<int64_t>(std::numeric_limits<InEigenType>::max());
                int64_t i32_min_in_64 = static_cast<int64_t>(std::numeric_limits<InEigenType>::min());
                REQUIRE(res_in_64 <= i32_max_in_64 && res_in_64 >= i32_min_in_64, "OpIntDiv: result not in i32 range");
                return static_cast<InEigenType>(res_in_64);
            };
            break;
        default:
            ERROR_IF(true, "unsupported TOSA_REF_TYPE %s", EnumNameTOSAREFTYPE(InDtype));
    }

    return 0;
}

template <int Rank, TOSA_REF_TYPE Dtype>
int OpLogicalAnd<Rank, Dtype>::register_fcn()
{
    switch (Dtype)
    {
        case TOSA_REF_TYPE_BOOL:
            this->fcn = [](InEigenType a, InEigenType b) -> OutEigenType { return a && b; };
            break;
        default:
            ERROR_IF(true, "unsupported TOSA_REF_TYPE %s", EnumNameTOSAREFTYPE(Dtype));
    }

    return 0;
}

template <int Rank, TOSA_REF_TYPE Dtype>
int OpLogicalLeftShift<Rank, Dtype>::register_fcn()
{
    switch (Dtype)
    {
        case TOSA_REF_TYPE_INT8:
            this->fcn = [this](InEigenType a, InEigenType b) -> OutEigenType {
                REQUIRE(b >= 0 && b <= 31, "OpLogicalLeftShift: shift value %d is out of valid range [0, 31]",
                        (int32_t)b);
                return static_cast<OutEigenType>(static_cast<int8_t>(a << b));
            };
            break;
        case TOSA_REF_TYPE_INT16:
            this->fcn = [this](InEigenType a, InEigenType b) -> OutEigenType {
                REQUIRE(b >= 0 && b <= 31, "OpLogicalLeftShift: shift value %d is out of valid range [0, 31]",
                        (int32_t)b);
                return static_cast<OutEigenType>(static_cast<int16_t>(a << b));
            };
            break;
        case TOSA_REF_TYPE_INT32:
            this->fcn = [this](InEigenType a, InEigenType b) -> OutEigenType {
                REQUIRE(b >= 0 && b <= 31, "OpLogicalLeftShift: shift value %d is out of valid range [0, 31]",
                        (int32_t)b);
                return static_cast<OutEigenType>(static_cast<int32_t>(a << b));
            };
            break;
        default:
            ERROR_IF(true, "unsupported TOSA_REF_TYPE %s", EnumNameTOSAREFTYPE(Dtype));
    }

    return 0;
}

template <int Rank, TOSA_REF_TYPE Dtype>
int OpLogicalRightShift<Rank, Dtype>::register_fcn()
{
    switch (Dtype)
    {
        case TOSA_REF_TYPE_INT8:
            this->fcn = [this](InEigenType a, InEigenType b) -> OutEigenType {
                REQUIRE(b >= 0 && b <= 31, "OpLogicalRightShift: shift value %d is out of valid range [0, 31]",
                        (int32_t)b);
                return static_cast<OutEigenType>(static_cast<int8_t>(static_cast<uint8_t>(a) >> b));
            };
            break;
        case TOSA_REF_TYPE_INT16:
            this->fcn = [this](InEigenType a, InEigenType b) -> OutEigenType {
                REQUIRE(b >= 0 && b <= 31, "OpLogicalRightShift: shift value %d is out of valid range [0, 31]",
                        (int32_t)b);
                return static_cast<OutEigenType>(static_cast<int16_t>(static_cast<uint16_t>(a) >> b));
            };
            break;
        case TOSA_REF_TYPE_INT32:
            this->fcn = [this](InEigenType a, InEigenType b) -> OutEigenType {
                REQUIRE(b >= 0 && b <= 31, "OpLogicalRightShift: shift value %d is out of valid range [0, 31]",
                        (int32_t)b);
                return static_cast<OutEigenType>(static_cast<int32_t>(static_cast<uint32_t>(a) >> b));
            };
            break;
        default:
            ERROR_IF(true, "unsupported TOSA_REF_TYPE %s", EnumNameTOSAREFTYPE(Dtype));
    }

    return 0;
}

template <int Rank, TOSA_REF_TYPE Dtype>
int OpLogicalOr<Rank, Dtype>::register_fcn()
{
    switch (Dtype)
    {
        case TOSA_REF_TYPE_BOOL:
            this->fcn = [](InEigenType a, InEigenType b) -> OutEigenType { return a || b; };
            break;
        default:
            ERROR_IF(true, "unsupported TOSA_REF_TYPE %s", EnumNameTOSAREFTYPE(Dtype));
    }

    return 0;
}

template <int Rank, TOSA_REF_TYPE Dtype>
int OpLogicalXor<Rank, Dtype>::register_fcn()
{
    switch (Dtype)
    {
        case TOSA_REF_TYPE_BOOL:
            this->fcn = [](InEigenType a, InEigenType b) -> OutEigenType { return a ^ b; };
            break;
        default:
            ERROR_IF(true, "unsupported TOSA_REF_TYPE %s", EnumNameTOSAREFTYPE(Dtype));
    }

    return 0;
}

template <int Rank, TOSA_REF_TYPE Dtype>
int OpMaximum<Rank, Dtype>::register_fcn()
{
    switch (Dtype)
    {
        case TOSA_REF_TYPE_FP16:
        case TOSA_REF_TYPE_BF16:
        case TOSA_REF_TYPE_FP32:
        case TOSA_REF_TYPE_FP64:
            this->fcn = [](InEigenType a, InEigenType b) -> OutEigenType {
                if (isnan(a))
                {
                    return a;
                }
                else if (isnan(b))
                {
                    return b;
                }
                else
                {
                    return a > b ? a : b;
                }
            };
            break;

        case TOSA_REF_TYPE_INT32:
            this->fcn = [](InEigenType a, InEigenType b) -> OutEigenType { return a > b ? a : b; };
            break;
        default:
            ERROR_IF(true, "unsupported TOSA_REF_TYPE %s", EnumNameTOSAREFTYPE(Dtype));
    }

    return 0;
}

template <int Rank, TOSA_REF_TYPE Dtype>
int OpMinimum<Rank, Dtype>::register_fcn()
{
    switch (Dtype)
    {
        case TOSA_REF_TYPE_FP16:
        case TOSA_REF_TYPE_BF16:
        case TOSA_REF_TYPE_FP32:
        case TOSA_REF_TYPE_FP64:
            this->fcn = [](InEigenType a, InEigenType b) -> OutEigenType {
                if (isnan(a))
                {
                    return a;
                }
                else if (isnan(b))
                {
                    return b;
                }
                else
                {
                    return a < b ? a : b;
                }
            };
            break;
        case TOSA_REF_TYPE_INT32:
            this->fcn = [](InEigenType a, InEigenType b) -> OutEigenType { return a < b ? a : b; };
            break;
        default:
            ERROR_IF(true, "unsupported TOSA_REF_TYPE %s", EnumNameTOSAREFTYPE(Dtype));
    }

    return 0;
}

template <int Rank, TOSA_REF_TYPE InDtype, TOSA_REF_TYPE OutDtype>
int OpMul<Rank, InDtype, OutDtype>::eval()
{
    // All cases except in_out_t == int32_t go to the general binary op workflow.
    if constexpr (InDtype != TOSA_REF_TYPE_INT32)
    {
        return BinaryNode<Rank, InDtype, OutDtype>::eval();
    }
    else
    {
        std::vector<int> calculated_shape;
        this->broadcast(calculated_shape);

        auto result_shape = this->result->getShape();
        ERROR_IF(calculated_shape != result_shape,
                 "Broadcast_shape failure, calculated_shape and result_shape don't match");

        TIn ia = this->a->getTensor().broadcast(this->bcast_a);
        TIn ib = this->b->getTensor().broadcast(this->bcast_b);

        using TInt64      = Eigen::Tensor<int64_t, Rank>;
        TInt64 tmp_result = ia.binaryExpr(ib, this->mul_fcn);

        // Retrieve `shift` value and construct a Eigen tensor instance for it. Shift is stored
        // as rank-0 tensor in Flatbuffer.
        auto s0 = dynamic_cast<TosaReference::TensorTemplate<TShiftRank0>*>(this->inputs[2]);

        // Get zero element from rank-0 tensor (i.e. shape = (0,)) in Numpy since `class Tensor`
        // currenly has no knowledge of the size of rank-0 tensor. Store rank-1 tensor instead
        // for testing.
        auto s1 = dynamic_cast<TosaReference::TensorTemplate<TShiftRank1>*>(this->inputs[2]);

        ASSERT_MEM(s0 || s1);

        int shift = s0 ? s0->getTensor()(0) : s1->getTensor()(0);
        TIn is(ia);
        is.setConstant(shift);

        TOut result               = tmp_result.binaryExpr(is, this->shr_fcn);
        this->result->getTensor() = result;

        return GraphNode::eval();
    }
}

// Eigen operators requires tensor operands meet NumDims > 0, partial specialize
// this like we did for the base class.
template <>
int OpMul<0, TOSA_REF_TYPE_INT32, TOSA_REF_TYPE_INT32>::eval()
{
    Eigen::Tensor<int64_t, 0> tmp_result = this->a->getTensor().binaryExpr(this->b->getTensor(), this->mul_fcn);

    // Retrieve `shift` value.
    auto s0 = dynamic_cast<TosaReference::TensorTemplate<TShiftRank0>*>(this->inputs[2]);
    auto s1 = dynamic_cast<TosaReference::TensorTemplate<TShiftRank1>*>(this->inputs[2]);
    ASSERT_MEM(s0 || s1);

    Eigen::Tensor<int64_t, 0> shift;
    shift.setConstant(s0 ? s0->getTensor()(0) : s1->getTensor()(0));

    this->result->getTensor() = tmp_result.binaryExpr(shift, this->shr_fcn);

    return GraphNode::eval();
}

template <int Rank, TOSA_REF_TYPE InDtype, TOSA_REF_TYPE OutDtype>
int OpMul<Rank, InDtype, OutDtype>::register_fcn()
{
    // Register evaluation function for in_out_t == int32_t case first as it supports shift
    // right to int32_t output.
    if constexpr (InDtype == TOSA_REF_TYPE_INT32)
    {
        // Perform multiplication on int32_t inputs to product int64_t result.
        this->mul_fcn = [](InEigenType a, InEigenType b) -> int64_t {
            int64_t result = static_cast<int64_t>(a) * static_cast<int64_t>(b);
            return result;
        };

        // Convert data from int64_t to int32_t.
        this->shr_fcn = [this](int64_t a, InEigenType shift) -> OutEigenType {
            int64_t result;
            if (shift > 0)
            {
                int64_t round = INT64_C(1) << (shift - 1);
                result        = a + round;
                result        = result >> shift;

                REQUIRE(result >= QMin && result <= QMax,
                        "OpMul: result %" PRId64 " exceeds valid range [%" PRId64 ", %" PRId64 "]", result, QMin, QMax);
            }
            else
            {
                result                = a;
                int64_t i32_max_in_64 = static_cast<int64_t>(std::numeric_limits<InEigenType>::max());
                int64_t i32_min_in_64 = static_cast<int64_t>(std::numeric_limits<InEigenType>::min());
                REQUIRE(result <= i32_max_in_64 && result >= i32_min_in_64, "OpMul: result not in i32 range");
                return static_cast<InEigenType>(result);
            }
            return static_cast<OutEigenType>(result);
        };

        return 0;
    }

    switch (InDtype)
    {
        case TOSA_REF_TYPE_FP16:
        case TOSA_REF_TYPE_BF16:
        case TOSA_REF_TYPE_FP32:
            this->fcn = [](InEigenType a, InEigenType b) -> OutEigenType { return fpTrunc<OutDtype>(a * b); };
            break;
        case TOSA_REF_TYPE_FP64:
            this->fcn = [](InEigenType a, InEigenType b) -> OutEigenType { return a * b; };
            break;
        case TOSA_REF_TYPE_INT8:
        case TOSA_REF_TYPE_INT16:
            this->fcn = [](InEigenType lhs, InEigenType rhs) -> OutEigenType {
                OutEigenType raw_output = (OutEigenType)lhs * (OutEigenType)rhs;

                OutEigenType clamped_output = std::min<OutEigenType>(QMax, std::max<OutEigenType>(raw_output, QMin));

                return clamped_output;
            };
            break;
        default:
            ERROR_IF(true, "unsupported TOSA_REF_TYPE %s", EnumNameTOSAREFTYPE(InDtype));
    }

    return 0;
}

template <int Rank, TOSA_REF_TYPE Dtype>
int OpPow<Rank, Dtype>::register_fcn()
{
    switch (Dtype)
    {
        case TOSA_REF_TYPE_FP16:
        case TOSA_REF_TYPE_BF16:
        case TOSA_REF_TYPE_FP32:
            this->fcn = [](InEigenType a, InEigenType b) -> OutEigenType { return fpTrunc<OutDtype>(powf(a, b)); };
            break;
        case TOSA_REF_TYPE_FP64:
            if (g_func_config.abs_mode)
            {
                // ABS_ERROR bounds return 2*(1+abs(log(abs(a))*b))
                this->fcn = [](InEigenType a, InEigenType b) -> OutEigenType {
                    OutEigenType c = log(a > (InEigenType)0 ? a : (-a)) * b;
                    return 2 * (1.0 + (c > (OutEigenType)0 ? c : (-c)));
                };
            }
            else
            {
                this->fcn = [](InEigenType a, InEigenType b) -> OutEigenType { return pow(a, b); };
            }
            break;
        default:
            ERROR_IF(true, "unsupported TOSA_REF_TYPE %s", EnumNameTOSAREFTYPE(Dtype));
    }

    return 0;
}

template <int Rank, TOSA_REF_TYPE Dtype>
int OpSub<Rank, Dtype>::register_fcn()
{
    switch (InDtype)
    {
        case TOSA_REF_TYPE_INT32:
            this->fcn = [this](InEigenType a, InEigenType b) -> OutEigenType {
                int64_t res_in_64     = static_cast<int64_t>(a) - b;
                int64_t i32_max_in_64 = static_cast<int64_t>(std::numeric_limits<InEigenType>::max());
                int64_t i32_min_in_64 = static_cast<int64_t>(std::numeric_limits<InEigenType>::min());
                REQUIRE(res_in_64 <= i32_max_in_64 && res_in_64 >= i32_min_in_64, "OpSub: result not in i32 range");
                return static_cast<InEigenType>(res_in_64);
            };
            break;
        case TOSA_REF_TYPE_FP16:
        case TOSA_REF_TYPE_BF16:
        case TOSA_REF_TYPE_FP32:
            this->fcn = [](InEigenType a, InEigenType b) -> OutEigenType { return fpTrunc<OutDtype>(a - b); };
            break;
        case TOSA_REF_TYPE_FP64:
            this->fcn = [](InEigenType a, InEigenType b) -> OutEigenType { return a - b; };
            break;
        default:
            ERROR_IF(true, "unsupported TOSA_REF_TYPE %s", EnumNameTOSAREFTYPE(InDtype));
    }

    return 0;
}

template <int Rank, TOSA_REF_TYPE InDtype>
OpTable<Rank, InDtype>::OpTable(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_)
    : GraphNode(sgt_, Op_TABLE, id_)
{
    setRequiredOperands(1, 1);
    setRequiredRank(0, 6);

    INIT_ATTRIBUTE(Table);
}

template <int Rank, TOSA_REF_TYPE InDtype>
OpTable<Rank, InDtype>::~OpTable()
{
    if (attribute)
        delete attribute;
}

template <int Rank, TOSA_REF_TYPE InDtype>
int OpTable<Rank, InDtype>::checkTensorAttributes()
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

    ERROR_IF(inputs[0]->getDtype() != InDtype, "OpTable: Unexpected input type");
    ERROR_IF(outputs[0]->getDtype() != OutDtype, "OpTable: Unexpected output type");
    ERROR_IF(attribute->table().size() != TableNumEntries, "OpTable: table attribute size must be %u", TableNumEntries);

    for (uint32_t i = 0; i < TableNumEntries; i++)
    {
        table[i] = (TableEigenType)attribute->table()[i];
    }

    in  = dynamic_cast<TosaReference::TensorTemplate<TIn>*>(inputs[0]);
    out = dynamic_cast<TosaReference::TensorTemplate<TOut>*>(outputs[0]);

    ASSERT_MEM(in && out);

    return 0;
}

template <int Rank, TOSA_REF_TYPE InDtype>
int OpTable<Rank, InDtype>::eval()
{
    switch (InDtype)
    {
        case TOSA_REF_TYPE_INT8:
            this->out->getTensor() = this->in->getTensor().unaryExpr([this](InEigenType in) -> OutEigenType {
                int32_t input_truncated = std::min<int32_t>(std::max<int32_t>(in, QInMin), QInMax);
                int32_t index           = input_truncated - QInMin;
                int32_t value           = table[index];

                return value;
            });
            break;
        case TOSA_REF_TYPE_INT16:
            this->out->getTensor() = this->in->getTensor().unaryExpr([this](InEigenType in) -> OutEigenType {
                // 1. make sure input is int16 range
                int32_t input_truncated = std::min<int32_t>(std::max<int32_t>(in, QInMin), QInMax);

                // 2. calculate index and interpolation fraction
                int32_t index = (input_truncated >> FractionBits) + (1 << (IntegerBits - 1));
                index         = std::min<int32_t>(std::max<int32_t>(index, 0), NumTableEntries - 1);    // 9-bit index
                int32_t frac  = (input_truncated)&0x7F;    // 7-bit fraction

                // 3. Add REQUIRE CHECK for extreme large/small slopes
                int32_t base  = table[index];
                int32_t next  = table[index + 1];
                int32_t slope = next - base;
                REQUIRE(slope <= std::numeric_limits<int16_t>::max() && slope >= std::numeric_limits<int16_t>::min(),
                        "OpTable: slope out of int16_t range");

                // 4. interpolate, generate 16.7 (23-bit) output
                int32_t value = (base << 7) + (slope)*frac;

                return value;
            });
            break;
        default:
            ERROR_IF(true, "unsupported TOSA_REF_TYPE %s", EnumNameTOSAREFTYPE(InDtype));
    }

    return GraphNode::eval();
}

// template explicit instantiation
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(BinaryNodeBase, FP16, FP16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(BinaryNodeBase, BF16, BF16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(BinaryNodeBase, FP32, FP32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(BinaryNodeBase, INT8, INT8);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(BinaryNodeBase, INT16, INT16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(BinaryNodeBase, INT32, INT32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(BinaryNodeBase, INT8, INT32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(BinaryNodeBase, INT16, INT32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(BinaryNodeBase, BOOL, BOOL);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(BinaryNodeBase, FP16, BOOL);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(BinaryNodeBase, BF16, BOOL);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(BinaryNodeBase, FP32, BOOL);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(BinaryNodeBase, INT32, BOOL);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(BinaryNodeBase, FP64, FP64);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(BinaryNodeBase, FP64, BOOL);

DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpAdd, FP16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpAdd, BF16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpAdd, FP32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpAdd, INT32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpAdd, FP64);

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

DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpIntdiv, INT32);

DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpLogicalAnd, BOOL);

DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpLogicalLeftShift, INT8);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpLogicalLeftShift, INT16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpLogicalLeftShift, INT32);

DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpLogicalRightShift, INT8);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpLogicalRightShift, INT16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpLogicalRightShift, INT32);

DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpLogicalOr, BOOL);

DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpLogicalXor, BOOL);

DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpMaximum, FP16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpMaximum, BF16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpMaximum, FP32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpMaximum, INT32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpMaximum, FP64);

DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpMinimum, FP16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpMinimum, BF16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpMinimum, FP32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpMinimum, INT32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpMinimum, FP64);

DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpMul, FP16, FP16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpMul, BF16, BF16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpMul, FP32, FP32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpMul, INT8, INT32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpMul, INT16, INT32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpMul, INT32, INT32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OpMul, FP64, FP64);

DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpPow, FP16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpPow, BF16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpPow, FP32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpPow, FP64);

DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpSub, FP16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpSub, BF16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpSub, FP32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpSub, INT32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpSub, FP64);

DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpTable, INT8);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpTable, INT16);

// Instantiation of nodes for comparison operators opEqual, opGreater
// and opGreaterEqual
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(BinaryNode, FP16, BOOL);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(BinaryNode, BF16, BOOL);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(BinaryNode, FP32, BOOL);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(BinaryNode, INT32, BOOL);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(BinaryNode, FP64, BOOL);

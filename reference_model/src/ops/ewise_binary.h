
// Copyright (c) 2020, ARM Limited.
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

#ifndef OPS_EWISE_BINARY_H
#define OPS_EWISE_BINARY_H

#include "graph_node.h"

using namespace tosa;

namespace TosaReference
{

// class BinaryNodeBase: hold common functions of all the binary nodes
//                       when an binary op is created, the virtual OpXXX::register_fcn() will be called
//                       and 'fcn' will be register with lambda function which has two inputs
// class BinaryNode: the level of indirection to partially specialize template for rank 0
//                   eval() from toplevel called should call the .binaryExpr(dims, fcn) here
//                   this needs to be partially specialize or
//                   compiler will statically fail when trying to broadcast rank0 tensor
// class OpXXX: implement per-element lambda function based on different data type
//              unlike BinaryNode, this doesn't need to be partially specialized

// Eigen::Tensor does support some binary element-wise natively (e.g. CWiseMax, or '+', etc.)
// which might be faster since it could be implemented with SIMD instructions
// the way of registering lambda + .binaryExpr() might sacrifice performance here
// but it can avoid partially specialization for combination of {rankN, rank0} x {FLOAT/INT32, QU8, ...}
// needs to revisit if performance becomes a bottleneck here
template <int Rank, DType InDtype, DType OutDtype>
class BinaryNodeBase : public GraphNode
{
public:
    BinaryNodeBase(const Op& nodeType, TosaQuantInfoBase* qinfo_, const uint64_t id_);
    virtual ~BinaryNodeBase();

    virtual int checkTensorAttributes() final;
    virtual int eval()         = 0;
    virtual int register_fcn() = 0;

    using InEigenType  = typename GetEigenType<InDtype>::type;
    using OutEigenType = typename GetEigenType<OutDtype>::type;
    using TIn          = Eigen::Tensor<InEigenType, Rank>;
    using TOut         = Eigen::Tensor<OutEigenType, Rank>;

protected:
    int broadcast();

protected:
    std::function<OutEigenType(InEigenType, InEigenType)> fcn;
    Eigen::array<int, Rank> bcast_a;
    Eigen::array<int, Rank> bcast_b;
    TosaReference::TensorTemplate<TIn>* a;
    TosaReference::TensorTemplate<TIn>* b;
    TosaReference::TensorTemplate<ETensor0<InEigenType>>* a_rank0;
    TosaReference::TensorTemplate<ETensor0<InEigenType>>* b_rank0;
    TosaReference::TensorTemplate<TOut>* result;
    int a_rank;
    int b_rank;
    int max_input_rank;
};

// primary class
template <int Rank, DType InDtype, DType OutDtype>
class BinaryNode : public BinaryNodeBase<Rank, InDtype, OutDtype>
{
public:
    BinaryNode(const Op& op_, TosaQuantInfoBase* qinfo_, const uint64_t id_)
        : BinaryNodeBase<Rank, InDtype, OutDtype>(op_, qinfo_, id_)
    {}
    virtual ~BinaryNode()
    {}

    virtual int eval();

    using InEigenType  = typename GetEigenType<InDtype>::type;
    using OutEigenType = typename GetEigenType<OutDtype>::type;
    using TIn          = Eigen::Tensor<InEigenType, Rank>;
    using TOut         = Eigen::Tensor<OutEigenType, Rank>;
};

// partial specialization for rank 0
template <DType InDtype, DType OutDtype>
class BinaryNode<0, InDtype, OutDtype> : public BinaryNodeBase<0, InDtype, OutDtype>
{
public:
    BinaryNode(const Op& op_, TosaQuantInfoBase* qinfo_, const uint64_t id_)
        : BinaryNodeBase<0, InDtype, OutDtype>(op_, qinfo_, id_)
    {}
    virtual ~BinaryNode()
    {}

    virtual int eval();
};

#define DEF_TEMPLATE_BINARY_OP_ONE_TYPE(Opname, OPNAME)                                                                \
    template <int Rank, DType Dtype>                                                                                   \
    class Op##Opname : public BinaryNode<Rank, Dtype, Dtype>                                                           \
    {                                                                                                                  \
    public:                                                                                                            \
        Op##Opname(TosaAttributeBase* attribute_, TosaQuantInfoBase* qinfo_, uint64_t id_)                             \
            : BinaryNode<Rank, Dtype, Dtype>(Op_##OPNAME, qinfo_, id_)                                                 \
        {                                                                                                              \
            register_fcn();                                                                                            \
        }                                                                                                              \
        static constexpr DType InDtype  = Dtype;                                                                       \
        static constexpr DType OutDtype = Dtype;                                                                       \
        using InEigenType               = typename GetEigenType<InDtype>::type;                                        \
        using OutEigenType              = typename GetEigenType<OutDtype>::type;                                       \
        virtual int register_fcn();                                                                                    \
    };

#define DEF_TEMPLATE_BINARY_OP_TWO_TYPE(Opname, OPNAME)                                                                \
    template <int Rank, DType InDtype, DType OutDtype>                                                                 \
    class Op##Opname : public BinaryNode<Rank, InDtype, OutDtype>                                                      \
    {                                                                                                                  \
    public:                                                                                                            \
        Op##Opname(TosaAttributeBase* attribute_, TosaQuantInfoBase* qinfo_, uint64_t id_)                             \
            : BinaryNode<Rank, InDtype, OutDtype>(Op_##OPNAME, qinfo_, id_)                                            \
        {                                                                                                              \
            register_fcn();                                                                                            \
        }                                                                                                              \
        static constexpr int32_t QMin = GetQMin<OutDtype>::value;                                                      \
        static constexpr int32_t QMax = GetQMax<OutDtype>::value;                                                      \
        using InEigenType             = typename GetEigenType<InDtype>::type;                                          \
        using OutEigenType            = typename GetEigenType<OutDtype>::type;                                         \
        virtual int register_fcn();                                                                                    \
    };

DEF_TEMPLATE_BINARY_OP_ONE_TYPE(Add, ADD)
DEF_TEMPLATE_BINARY_OP_ONE_TYPE(ArithmeticRightShift, ARITHMETIC_RIGHT_SHIFT)
DEF_TEMPLATE_BINARY_OP_ONE_TYPE(BitwiseAnd, BITWISE_AND)
DEF_TEMPLATE_BINARY_OP_ONE_TYPE(BitwiseOr, BITWISE_OR)
DEF_TEMPLATE_BINARY_OP_ONE_TYPE(BitwiseXor, BITWISE_XOR)
DEF_TEMPLATE_BINARY_OP_ONE_TYPE(LogicalAnd, LOGICAL_AND)
DEF_TEMPLATE_BINARY_OP_ONE_TYPE(LogicalLeftShift, LOGICAL_LEFT_SHIFT)
DEF_TEMPLATE_BINARY_OP_ONE_TYPE(LogicalRightShift, LOGICAL_RIGHT_SHIFT)
DEF_TEMPLATE_BINARY_OP_ONE_TYPE(LogicalOr, LOGICAL_OR)
DEF_TEMPLATE_BINARY_OP_ONE_TYPE(LogicalXor, LOGICAL_XOR)
DEF_TEMPLATE_BINARY_OP_ONE_TYPE(Maximum, MAXIMUM)
DEF_TEMPLATE_BINARY_OP_ONE_TYPE(Minimum, MINIMUM)
DEF_TEMPLATE_BINARY_OP_TWO_TYPE(Mul, MUL)
DEF_TEMPLATE_BINARY_OP_ONE_TYPE(Pow, POW)
DEF_TEMPLATE_BINARY_OP_ONE_TYPE(Sub, SUB)

#undef DEF_TEMPLATE_BINARY_OP_ONE_TYPE
#undef DEF_TEMPLATE_BINARY_OP_TWO_TYPE

template <int Rank>
class OpTable : public GraphNode
{
public:
    OpTable(TosaAttributeBase* attribute_, TosaQuantInfoBase* qinfo_, uint64_t id_);
    virtual ~OpTable();

    virtual int checkTensorAttributes();
    virtual int eval();

    static constexpr DType InDtype           = DType_INT16;
    static constexpr DType TableDtype        = DType_INT16;
    static constexpr DType OutDtype          = DType_INT32;
    using InEigenType                        = typename GetEigenType<InDtype>::type;
    using TableEigenType                     = typename GetEigenType<TableDtype>::type;
    using OutEigenType                       = typename GetEigenType<OutDtype>::type;
    using TIn                                = Eigen::Tensor<InEigenType, Rank>;
    using TTable                             = Eigen::Tensor<TableEigenType, 1>;
    using TOut                               = Eigen::Tensor<OutEigenType, Rank>;
    static constexpr int32_t IntegerBits     = 9;
    static constexpr int32_t FractionBits    = 7;
    static constexpr int32_t NumTableEntries = (1 << IntegerBits);
    static constexpr int32_t QInMin          = GetQMin<InDtype>::value;
    static constexpr int32_t QInMax          = GetQMax<InDtype>::value;
    static constexpr int32_t QOutMin         = GetQMin<OutDtype>::value;
    static constexpr int32_t QOutMax         = GetQMax<OutDtype>::value;

protected:
    TosaReference::TensorTemplate<TIn>* in;
    TosaReference::TensorTemplate<TTable>* table;
    TosaReference::TensorTemplate<TOut>* out;
};

};    // namespace TosaReference

#endif

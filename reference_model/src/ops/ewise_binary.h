
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
    BinaryNodeBase(SubgraphTraverser* sgt_, const Op& nodeType, const uint64_t id_);
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
    TosaReference::TensorTemplate<TOut>* result;
};

// primary class
template <int Rank, DType InDtype, DType OutDtype>
class BinaryNode : public BinaryNodeBase<Rank, InDtype, OutDtype>
{
public:
    BinaryNode(SubgraphTraverser* sgt_, const Op& op_, const uint64_t id_)
        : BinaryNodeBase<Rank, InDtype, OutDtype>(sgt_, op_, id_)
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
    BinaryNode(SubgraphTraverser* sgt_, const Op& op_, const uint64_t id_)
        : BinaryNodeBase<0, InDtype, OutDtype>(sgt_, op_, id_)
    {}
    virtual ~BinaryNode()
    {}

    virtual int eval();
};

#define DEF_TEMPLATE_BINARY_OP_DEFAULT(Opname, OPNAME)                                                                 \
    template <int Rank, DType Dtype>                                                                                   \
    class Op##Opname : public BinaryNode<Rank, Dtype, Dtype>                                                           \
    {                                                                                                                  \
    public:                                                                                                            \
        Op##Opname(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_)    \
            : BinaryNode<Rank, Dtype, Dtype>(sgt_, Op_##OPNAME, id_)                                           \
        {                                                                                                              \
            register_fcn();                                                                                            \
        }                                                                                                              \
        static constexpr DType InDtype  = Dtype;                                                                       \
        static constexpr DType OutDtype = Dtype;                                                                       \
        using InEigenType               = typename GetEigenType<InDtype>::type;                                        \
        using OutEigenType              = typename GetEigenType<OutDtype>::type;                                       \
        virtual int register_fcn();                                                                                    \
    };

DEF_TEMPLATE_BINARY_OP_DEFAULT(Add, ADD)
DEF_TEMPLATE_BINARY_OP_DEFAULT(BitwiseAnd, BITWISE_AND)
DEF_TEMPLATE_BINARY_OP_DEFAULT(BitwiseOr, BITWISE_OR)
DEF_TEMPLATE_BINARY_OP_DEFAULT(BitwiseXor, BITWISE_XOR)
DEF_TEMPLATE_BINARY_OP_DEFAULT(Intdiv, INTDIV)
DEF_TEMPLATE_BINARY_OP_DEFAULT(LogicalAnd, LOGICAL_AND)
DEF_TEMPLATE_BINARY_OP_DEFAULT(LogicalLeftShift, LOGICAL_LEFT_SHIFT)
DEF_TEMPLATE_BINARY_OP_DEFAULT(LogicalRightShift, LOGICAL_RIGHT_SHIFT)
DEF_TEMPLATE_BINARY_OP_DEFAULT(LogicalOr, LOGICAL_OR)
DEF_TEMPLATE_BINARY_OP_DEFAULT(LogicalXor, LOGICAL_XOR)
DEF_TEMPLATE_BINARY_OP_DEFAULT(Maximum, MAXIMUM)
DEF_TEMPLATE_BINARY_OP_DEFAULT(Minimum, MINIMUM)
DEF_TEMPLATE_BINARY_OP_DEFAULT(Pow, POW)
DEF_TEMPLATE_BINARY_OP_DEFAULT(Sub, SUB)

#undef DEF_TEMPLATE_BINARY_OP_DEFAULT

template <int Rank, DType Dtype>
class OpArithmeticRightShift : public BinaryNode<Rank, Dtype, Dtype>
{
public:
    OpArithmeticRightShift(SubgraphTraverser* sgt_,
                           TosaAttributeBase* attribute_,
                           uint64_t id_)
        : BinaryNode<Rank, Dtype, Dtype>(sgt_, Op_ARITHMETIC_RIGHT_SHIFT, id_)
    {
        INIT_ATTRIBUTE(ArithmeticRightShift);
        register_fcn();
    }
    using InEigenType  = typename GetEigenType<Dtype>::type;
    using OutEigenType = typename GetEigenType<Dtype>::type;
    virtual int register_fcn();

protected:
    TosaArithmeticRightShiftAttribute* attribute;
};

template <int Rank, DType InDtype, DType OutDtype>
class OpMul : public BinaryNode<Rank, InDtype, OutDtype>
{
public:
    OpMul(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_)
        : BinaryNode<Rank, InDtype, OutDtype>(sgt_, Op_MUL, id_)
    {
        INIT_ATTRIBUTE(Mul);
        register_fcn();
    }
    static constexpr int64_t QMin = GetQMin<OutDtype>::value;
    static constexpr int64_t QMax = GetQMax<OutDtype>::value;
    using InEigenType             = typename GetEigenType<InDtype>::type;
    using OutEigenType            = typename GetEigenType<OutDtype>::type;
    virtual int register_fcn();

protected:
    TosaMulAttribute* attribute;
};

template <int Rank, DType InDtype>
class OpTable : public GraphNode
{
public:
    OpTable(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_);
    virtual ~OpTable();

    virtual int checkTensorAttributes();
    virtual int eval();

    static constexpr DType TableDtype         = (InDtype == DType_INT8) ? DType_INT8 : DType_INT16;
    static constexpr DType OutDtype           = (InDtype == DType_INT8) ? DType_INT8 : DType_INT32;
    static constexpr uint32_t TableNumEntries = (InDtype == DType_INT8) ? 256 : 513;
    using InEigenType                         = typename GetEigenType<InDtype>::type;
    using TableEigenType                      = typename GetEigenType<TableDtype>::type;
    using OutEigenType                        = typename GetEigenType<OutDtype>::type;
    using TIn                                 = Eigen::Tensor<InEigenType, Rank>;
    using TTable                              = Eigen::Tensor<TableEigenType, 1>;
    using TOut                                = Eigen::Tensor<OutEigenType, Rank>;
    static constexpr int32_t IntegerBits      = 9;
    static constexpr int32_t FractionBits     = 7;
    static constexpr int32_t NumTableEntries  = (1 << IntegerBits);
    static constexpr int32_t QInMin           = GetQMin<InDtype>::value;
    static constexpr int32_t QInMax           = GetQMax<InDtype>::value;
    static constexpr int32_t QOutMin          = GetQMin<OutDtype>::value;
    static constexpr int32_t QOutMax          = GetQMax<OutDtype>::value;

protected:
    TosaReference::TensorTemplate<TIn>* in;
    TosaReference::TensorTemplate<TOut>* out;
    TosaTableAttribute* attribute;
    std::array<TableEigenType, TableNumEntries> table;
};

};    // namespace TosaReference

#endif

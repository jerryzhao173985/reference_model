
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

#ifndef OPS_EWISE_UNARY_H
#define OPS_EWISE_UNARY_H

#include "graph_node.h"

using namespace tosa;

namespace TosaReference
{
template <int Rank, DType Dtype>
class UnaryNode : public GraphNode
{
public:
    UnaryNode(SubgraphTraverser* sgt_, const Op& nodeType, const uint64_t id_);
    virtual ~UnaryNode();

    virtual int checkTensorAttributes() final;
    virtual int eval() final;
    virtual int register_fcn() = 0;

    using InEigenType  = typename GetEigenType<Dtype>::type;
    using OutEigenType = typename GetEigenType<Dtype>::type;
    using TIn          = Eigen::Tensor<InEigenType, Rank>;
    using TOut         = Eigen::Tensor<OutEigenType, Rank>;

protected:
    std::function<OutEigenType(InEigenType)> fcn;
    TosaReference::TensorTemplate<TIn>* a;
    TosaReference::TensorTemplate<TOut>* result;
};

#define DEF_TEMPLATE_UNARY_OP(Opname, OPNAME)                                                                          \
    template <int Rank, DType Dtype>                                                                                   \
    class Op##Opname : public UnaryNode<Rank, Dtype>                                                                   \
    {                                                                                                                  \
    public:                                                                                                            \
        Op##Opname(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, TosaQuantInfoBase* qinfo_, uint64_t id_)    \
            : UnaryNode<Rank, Dtype>(sgt_, Op_##OPNAME, id_)                                                           \
        {                                                                                                              \
            register_fcn();                                                                                            \
        }                                                                                                              \
        static constexpr int32_t QMin = GetQMin<Dtype>::value;                                                         \
        static constexpr int32_t QMax = GetQMax<Dtype>::value;                                                         \
        using InEigenType             = typename GetEigenType<Dtype>::type;                                            \
        using OutEigenType            = typename GetEigenType<Dtype>::type;                                            \
        virtual int register_fcn();                                                                                    \
    };

#define DEF_TEMPLATE_UNARY_OP_WITH_QUANT_INFO(Opname, OPNAME)                                                          \
    template <int Rank, DType Dtype>                                                                                   \
    class Op##Opname : public UnaryNode<Rank, Dtype>                                                                   \
    {                                                                                                                  \
    public:                                                                                                            \
        Op##Opname(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, TosaQuantInfoBase* qinfo_, uint64_t id_)    \
            : UnaryNode<Rank, Dtype>(sgt_, Op_##OPNAME, id_)                                                           \
        {                                                                                                              \
            INIT_QINFO(Unary);                                                                                         \
            register_fcn();                                                                                            \
        }                                                                                                              \
        static constexpr int32_t QMin = GetQMin<Dtype>::value;                                                         \
        static constexpr int32_t QMax = GetQMax<Dtype>::value;                                                         \
        using InEigenType             = typename GetEigenType<Dtype>::type;                                            \
        using OutEigenType            = typename GetEigenType<Dtype>::type;                                            \
        virtual int register_fcn();                                                                                    \
                                                                                                                       \
    protected:                                                                                                         \
        TosaUnaryQuantInfo* qinfo;                                                                                     \
    };

DEF_TEMPLATE_UNARY_OP(Abs, ABS)
DEF_TEMPLATE_UNARY_OP(BitwiseNot, BITWISE_NOT)
DEF_TEMPLATE_UNARY_OP(Ceil, CEIL)
DEF_TEMPLATE_UNARY_OP(Clz, CLZ)
DEF_TEMPLATE_UNARY_OP(Exp, EXP)
DEF_TEMPLATE_UNARY_OP(Floor, FLOOR)
DEF_TEMPLATE_UNARY_OP(Log, LOG)
DEF_TEMPLATE_UNARY_OP(LogicalNot, LOGICAL_NOT)
DEF_TEMPLATE_UNARY_OP_WITH_QUANT_INFO(Negate, NEGATE)
DEF_TEMPLATE_UNARY_OP(Reciprocal, RECIPROCAL)
DEF_TEMPLATE_UNARY_OP(Rsqrt, RSQRT)

#undef DEF_TEMPLATE_UNARY_OP
#undef DEF_TEMPLATE_UNARY_OP_WITH_QUANT_INFO

};    // namespace TosaReference

#endif

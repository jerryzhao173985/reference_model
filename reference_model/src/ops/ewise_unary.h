
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

#ifndef OPS_EWISE_UNARY_H
#define OPS_EWISE_UNARY_H

#include "graph_node.h"

using namespace tosa;

namespace TosaReference
{
template <int Rank, TOSA_REF_TYPE Dtype>
class UnaryNode : public GraphNode
{
public:
    UnaryNode(SubgraphTraverser* sgt_, const Op& nodeType, const uint64_t id_);
    virtual ~UnaryNode();

    virtual int checkTensorAttributes();
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
    template <int Rank, TOSA_REF_TYPE Dtype>                                                                           \
    class Op##Opname : public UnaryNode<Rank, Dtype>                                                                   \
    {                                                                                                                  \
    public:                                                                                                            \
        Op##Opname(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_)                               \
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

DEF_TEMPLATE_UNARY_OP(Abs, ABS)
DEF_TEMPLATE_UNARY_OP(BitwiseNot, BITWISE_NOT)
DEF_TEMPLATE_UNARY_OP(Ceil, CEIL)
DEF_TEMPLATE_UNARY_OP(Clz, CLZ)
DEF_TEMPLATE_UNARY_OP(Cos, COS)
DEF_TEMPLATE_UNARY_OP(Exp, EXP)
DEF_TEMPLATE_UNARY_OP(Floor, FLOOR)
DEF_TEMPLATE_UNARY_OP(Log, LOG)
DEF_TEMPLATE_UNARY_OP(LogicalNot, LOGICAL_NOT)
DEF_TEMPLATE_UNARY_OP(Reciprocal, RECIPROCAL)
DEF_TEMPLATE_UNARY_OP(Rsqrt, RSQRT)
DEF_TEMPLATE_UNARY_OP(Sin, SIN)

#undef DEF_TEMPLATE_UNARY_OP

// Negate is the only unary op with attributes
template <int Rank, TOSA_REF_TYPE Dtype>
class OpNegate : public UnaryNode<Rank, Dtype>
{
public:
    OpNegate(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_);
    virtual ~OpNegate();

    static constexpr int32_t QMin = GetQMin<Dtype>::value;
    static constexpr int32_t QMax = GetQMax<Dtype>::value;
    using InEigenType             = typename GetEigenType<Dtype>::type;
    using OutEigenType            = typename GetEigenType<Dtype>::type;
    virtual int register_fcn();

protected:
    std::unique_ptr<tosa::TosaNegateAttribute> attribute;
};

};    // namespace TosaReference

#endif

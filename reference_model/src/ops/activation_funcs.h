
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

#ifndef OPS_ACTIVATION_FUNCS_H
#define OPS_ACTIVATION_FUNCS_H

#include "ewise_unary.h"
#include "graph_node.h"

using namespace tosa;

namespace TosaReference
{

template <int Rank, TOSA_REF_TYPE Dtype>
class OpClamp : public UnaryNode<Rank, Dtype>
{
public:
    OpClamp(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_)
        : UnaryNode<Rank, Dtype>(sgt_, Op_CLAMP, id_)
    {
        INIT_ATTRIBUTE(Clamp);
    }
    virtual ~OpClamp();
    static constexpr int32_t QMin = GetQMin<Dtype>::value;
    static constexpr int32_t QMax = GetQMax<Dtype>::value;
    using InEigenType             = typename GetEigenType<Dtype>::type;
    using OutEigenType            = typename GetEigenType<Dtype>::type;
    virtual int register_fcn();
    virtual int checkTensorAttributes();

protected:
    std::unique_ptr<TosaClampAttribute> attribute;
};

template <int Rank, TOSA_REF_TYPE Dtype>
class OpSigmoid : public UnaryNode<Rank, Dtype>
{
public:
    OpSigmoid(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_)
        : UnaryNode<Rank, Dtype>(sgt_, Op_SIGMOID, id_)
    {
        register_fcn();
    }
    static constexpr int32_t QMin = GetQMin<Dtype>::value;
    static constexpr int32_t QMax = GetQMax<Dtype>::value;
    using InEigenType             = typename GetEigenType<Dtype>::type;
    using OutEigenType            = typename GetEigenType<Dtype>::type;
    virtual int register_fcn();
};

template <int Rank, TOSA_REF_TYPE Dtype>
class OpTanh : public UnaryNode<Rank, Dtype>
{
public:
    OpTanh(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_)
        : UnaryNode<Rank, Dtype>(sgt_, Op_TANH, id_)
    {
        register_fcn();
    }
    static constexpr int32_t QMin = GetQMin<Dtype>::value;
    static constexpr int32_t QMax = GetQMax<Dtype>::value;
    using InEigenType             = typename GetEigenType<Dtype>::type;
    using OutEigenType            = typename GetEigenType<Dtype>::type;
    virtual int register_fcn();
};

template <int Rank, TOSA_REF_TYPE Dtype>
class OpErf : public UnaryNode<Rank, Dtype>
{
public:
    OpErf(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_)
        : UnaryNode<Rank, Dtype>(sgt_, Op_ERF, id_)
    {
        register_fcn();
    }
    using InEigenType  = typename GetEigenType<Dtype>::type;
    using OutEigenType = typename GetEigenType<Dtype>::type;
    virtual int register_fcn();
};

};    // namespace TosaReference

#endif

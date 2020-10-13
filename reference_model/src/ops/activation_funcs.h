
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

#ifndef OPS_ACTIVATION_FUNCS_H
#define OPS_ACTIVATION_FUNCS_H

#include "ewise_unary.h"
#include "graph_node.h"

using namespace tosa;

namespace TosaReference
{

template <int Rank, DType Dtype>
class OpClamp : public UnaryNode<Rank, Dtype>
{
public:
    OpClamp(TosaAttributeBase* attribute_, TosaQuantInfoBase* qinfo_, uint64_t id_)
        : UnaryNode<Rank, Dtype>(Op_CLAMP, id_)
    {
        INIT_ATTRIBUTE(Clamp);
        register_fcn();
    }
    static constexpr int32_t QMin = GetQMin<Dtype>::value;
    static constexpr int32_t QMax = GetQMax<Dtype>::value;
    using InEigenType             = typename GetEigenType<Dtype>::type;
    using OutEigenType            = typename GetEigenType<Dtype>::type;
    virtual int register_fcn();

protected:
    TosaClampAttribute* attribute;
};

template <int Rank, DType Dtype>
class OpReluN : public UnaryNode<Rank, Dtype>
{
public:
    OpReluN(TosaAttributeBase* attribute_, TosaQuantInfoBase* qinfo_, uint64_t id_)
        : UnaryNode<Rank, Dtype>(Op_RELUN, id_)
    {
        INIT_ATTRIBUTE(ReluN);
        register_fcn();
    }
    static constexpr int32_t QMin = GetQMin<Dtype>::value;
    static constexpr int32_t QMax = GetQMax<Dtype>::value;
    using InEigenType             = typename GetEigenType<Dtype>::type;
    using OutEigenType            = typename GetEigenType<Dtype>::type;
    virtual int register_fcn();

protected:
    TosaReluNAttribute* attribute;
};

template <int Rank, DType Dtype>
class OpSigmoid : public UnaryNode<Rank, Dtype>
{
public:
    OpSigmoid(TosaAttributeBase* attribute_, TosaQuantInfoBase* qinfo_, uint64_t id_)
        : UnaryNode<Rank, Dtype>(Op_SIGMOID, id_)
    {
        register_fcn();
    }
    static constexpr int32_t QMin = GetQMin<Dtype>::value;
    static constexpr int32_t QMax = GetQMax<Dtype>::value;
    using InEigenType             = typename GetEigenType<Dtype>::type;
    using OutEigenType            = typename GetEigenType<Dtype>::type;
    virtual int register_fcn();
};

template <int Rank, DType Dtype>
class OpTanh : public UnaryNode<Rank, Dtype>
{
public:
    OpTanh(TosaAttributeBase* attribute_, TosaQuantInfoBase* qinfo_, uint64_t id_)
        : UnaryNode<Rank, Dtype>(Op_TANH, id_)
    {
        register_fcn();
    }
    static constexpr int32_t QMin = GetQMin<Dtype>::value;
    static constexpr int32_t QMax = GetQMax<Dtype>::value;
    using InEigenType             = typename GetEigenType<Dtype>::type;
    using OutEigenType            = typename GetEigenType<Dtype>::type;
    virtual int register_fcn();
};

};    // namespace TosaReference

#endif

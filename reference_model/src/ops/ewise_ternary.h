
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

#ifndef OPS_TERNARY_H
#define OPS_TERNARY_H

#include "graph_node.h"

using namespace tosa;

namespace TosaReference
{

// The Ternary Select op has the following operands:
//   1.   Cond:   rank N, type=bool
//   2.   Then_val: Rank N, type=<V>
//   3.   Else_val: Rank N, type=<V>
//   4.   Result:   Rank N, type=<V>
// Cond, Then_val, Else_val need to be mutually-broadcastable
template <int Rank, DType Dtype>
class OpSelectBase : public GraphNode
{
public:
    OpSelectBase(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_);
    virtual ~OpSelectBase();

    virtual int checkTensorAttributes();
    virtual int eval();

    using CondEigenType = typename GetEigenType<DType_BOOL>::type;
    using InEigenType   = typename GetEigenType<Dtype>::type;
    using TCond         = Eigen::Tensor<CondEigenType, Rank>;
    using TIn           = Eigen::Tensor<InEigenType, Rank>;

protected:
    TosaReference::TensorTemplate<TCond>* cond;
    Eigen::array<int, Rank> bcast_cond;
    Eigen::array<int, Rank> bcast_then;
    Eigen::array<int, Rank> bcast_else;
    TosaReference::TensorTemplate<TIn>* then_val;
    TosaReference::TensorTemplate<TIn>* else_val;
    TosaReference::TensorTemplate<TIn>* out;
};

// primary class
template <int Rank, DType Dtype>
class OpSelect : public OpSelectBase<Rank, Dtype>
{
public:
    OpSelect(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_)
        : OpSelectBase<Rank, Dtype>(sgt_, attribute_, id_)
    {}
    virtual int eval();
    int broadcast();

    using InEigenType = typename OpSelectBase<Rank, Dtype>::InEigenType;
};

// partial specialization for rank 0
template <DType Dtype>
class OpSelect<0, Dtype> : public OpSelectBase<0, Dtype>
{
public:
    OpSelect(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_)
        : OpSelectBase<0, Dtype>(sgt_, attribute_, id_)
    {}
    virtual int eval();
};
};    // namespace TosaReference

#endif

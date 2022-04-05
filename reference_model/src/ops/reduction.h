
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

#ifndef OPS_REDUCTION_H
#define OPS_REDUCTION_H

#include "graph_node.h"

using namespace tosa;

namespace TosaReference
{

template <int Rank, DType Dtype>
class ReduceNode : public GraphNode
{
public:
    ReduceNode(SubgraphTraverser* sgt_, const Op& nodeType, TosaAttributeBase* attribute_, const uint64_t id_);
    virtual ~ReduceNode();
    virtual int checkTensorAttributes();
    virtual int eval() = 0;

    using InEigenType  = typename GetEigenType<Dtype>::type;
    using OutEigenType = typename GetEigenType<Dtype>::type;
    using TIn          = Eigen::Tensor<InEigenType, Rank>;
    using TOut         = Eigen::Tensor<OutEigenType, Rank>;

protected:
    Eigen::array<int, 1> dims;
    TosaReference::TensorTemplate<TIn>* in;
    TosaReference::TensorTemplate<TOut>* out;
    TosaAxisAttribute* attribute;
};

template <int Rank, DType Dtype>
class OpReduceAll : public ReduceNode<Rank, Dtype>
{
public:
    OpReduceAll(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, TosaQuantInfoBase* qinfo_, uint64_t id_)
        : ReduceNode<Rank, Dtype>(sgt_, Op_REDUCE_ALL, attribute_, id_)
    {}
    virtual int eval();
};

template <int Rank, DType Dtype>
class OpReduceAny : public ReduceNode<Rank, Dtype>
{
public:
    OpReduceAny(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, TosaQuantInfoBase* qinfo_, uint64_t id_)
        : ReduceNode<Rank, Dtype>(sgt_, Op_REDUCE_ALL, attribute_, id_)
    {}
    virtual int eval();
};

template <int Rank, DType Dtype>
class OpReduceMax : public ReduceNode<Rank, Dtype>
{
public:
    OpReduceMax(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, TosaQuantInfoBase* qinfo_, uint64_t id_)
        : ReduceNode<Rank, Dtype>(sgt_, Op_REDUCE_MAX, attribute_, id_)
    {}
    virtual int eval();
};

template <int Rank, DType Dtype>
class OpReduceMin : public ReduceNode<Rank, Dtype>
{
public:
    OpReduceMin(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, TosaQuantInfoBase* qinfo_, uint64_t id_)
        : ReduceNode<Rank, Dtype>(sgt_, Op_REDUCE_MIN, attribute_, id_)
    {}
    virtual int eval();
};

template <int Rank, DType Dtype>
class OpReduceProduct : public ReduceNode<Rank, Dtype>
{
public:
    OpReduceProduct(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, TosaQuantInfoBase* qinfo_, uint64_t id_)
        : ReduceNode<Rank, Dtype>(sgt_, Op_REDUCE_PRODUCT, attribute_, id_)
    {}
    virtual int eval();
};

template <int Rank, DType Dtype>
class OpReduceSum : public ReduceNode<Rank, Dtype>
{
public:
    OpReduceSum(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, TosaQuantInfoBase* qinfo_, uint64_t id_)
        : ReduceNode<Rank, Dtype>(sgt_, Op_REDUCE_SUM, attribute_, id_)
    {}
    virtual int eval();
};

template <int Rank, DType Dtype>
class OpReduceSumInt : public ReduceNode<Rank, Dtype>
{
public:
    OpReduceSumInt(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, TosaQuantInfoBase* qinfo_, uint64_t id_)
        : ReduceNode<Rank, Dtype>(sgt_, Op_REDUCE_SUM, attribute_, id_)
    {}
    virtual int eval();
};

};    // namespace TosaReference

#endif


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

#ifndef OPS_SCATTER_GATHER_H
#define OPS_SCATTER_GATHER_H

#include "graph_node.h"

using namespace tosa;

namespace TosaReference
{

template <DType Dtype>
class OpGather : public GraphNode
{
public:
    OpGather(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, TosaQuantInfoBase* qinfo_, uint64_t id_);
    virtual ~OpGather();

    virtual int checkTensorAttributes();
    virtual int eval();

    using EigenType = typename GetEigenType<Dtype>::type;
    using TValue    = Eigen::Tensor<EigenType, 3>;
    using TIndex    = Eigen::Tensor<int32_t, 2>;
    using TOutput   = Eigen::Tensor<EigenType, 3>;

protected:
    int32_t N, W, K, C;
    TosaReference::TensorTemplate<TValue>* values;
    TosaReference::TensorTemplate<TIndex>* indices;
    TosaReference::TensorTemplate<TOutput>* output;
};

template <DType Dtype>
class OpScatter : public GraphNode
{
public:
    OpScatter(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, TosaQuantInfoBase* qinfo_, uint64_t id_);
    virtual ~OpScatter();

    virtual int checkTensorAttributes();
    virtual int eval();

    using EigenType = typename GetEigenType<Dtype>::type;
    using TValue    = Eigen::Tensor<EigenType, 3>;
    using TIndex    = Eigen::Tensor<int32_t, 2>;
    using TOutput   = Eigen::Tensor<EigenType, 3>;

protected:
    int32_t N, W, K, C;
    TosaReference::TensorTemplate<TValue>* values_in;
    TosaReference::TensorTemplate<TIndex>* indices;
    TosaReference::TensorTemplate<TValue>* input;
    TosaReference::TensorTemplate<TOutput>* values_out;
};

};    // namespace TosaReference

#endif

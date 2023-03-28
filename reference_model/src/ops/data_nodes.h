
// Copyright (c) 2020-2023, ARM Limited.
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

#ifndef OPS_DATA_NODES_H
#define OPS_DATA_NODES_H

#include "graph_node.h"

namespace TosaReference
{

class OpConst : public GraphNode
{
public:
    OpConst(SubgraphTraverser* sgt_, uint64_t id_);
    virtual ~OpConst();

    virtual int checkTensorAttributes();
    virtual int eval();
};

template <int Rank, TOSA_REF_TYPE Dtype>
class OpIdentity : public GraphNode
{
public:
    OpIdentity(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_);
    virtual ~OpIdentity();

    virtual int checkTensorAttributes();
    virtual int eval();

    using InEigenType  = typename GetEigenType<Dtype>::type;
    using OutEigenType = typename GetEigenType<Dtype>::type;
    using TIn          = Eigen::Tensor<InEigenType, Rank>;
    using TOut         = Eigen::Tensor<OutEigenType, Rank>;

protected:
    TosaReference::TensorTemplate<TIn>* in;
    TosaReference::TensorTemplate<TOut>* out;
};

};    // namespace TosaReference

#endif

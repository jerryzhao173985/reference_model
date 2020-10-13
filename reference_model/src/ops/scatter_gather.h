
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

// input and index can have different rank
// and infer OutRank statically
template <int InRank, int IndexRank, DType Dtype>
class OpGather : public GraphNode
{
public:
    OpGather(TosaAttributeBase* attribute_, TosaQuantInfoBase* qinfo_, uint64_t id_);
    virtual ~OpGather();

    virtual int checkTensorAttributes();
    virtual int eval();

    static constexpr int OutRank = InRank - 1 + IndexRank;
    using InEigenType            = typename GetEigenType<Dtype>::type;
    using OutEigenType           = typename GetEigenType<Dtype>::type;
    using TIn                    = Eigen::Tensor<InEigenType, InRank>;
    using TIndex                 = Eigen::Tensor<int32_t, IndexRank>;
    using TOut                   = Eigen::Tensor<OutEigenType, OutRank>;

protected:
    TosaAxisAttribute* attribute;
    TosaReference::TensorTemplate<TIn>* in;
    TosaReference::TensorTemplate<TIndex>* index;
    TosaReference::TensorTemplate<TOut>* out;
};

};    // namespace TosaReference

#endif

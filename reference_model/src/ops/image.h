
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

#ifndef OPS_IMAGE_H
#define OPS_IMAGE_H

#include "graph_node.h"

using namespace tosa;

namespace TosaReference
{

template <DType InDtype, DType OutDtype>
class OpResize : public GraphNode
{
public:
    OpResize(TosaAttributeBase* attribute_, TosaQuantInfoBase* qinfo_, uint64_t id_);
    virtual ~OpResize();
    virtual int checkTensorAttributes() final;
    virtual int eval();

    using InEigenType  = typename GetEigenType<InDtype>::type;
    using OutEigenType = typename GetEigenType<OutDtype>::type;
    using TIn          = Eigen::Tensor<InEigenType, 4>;
    using TOut         = Eigen::Tensor<OutEigenType, 4>;

protected:
    TosaResizeAttribute* attribute;
    std::vector<int32_t> output_size;
    std::vector<int32_t> stride;
    std::vector<int32_t> offset;
    int32_t shift;
    ResizeMode mode;
    TosaReference::TensorTemplate<TIn>* in;
    TosaReference::TensorTemplate<TOut>* out;
};

};    // namespace TosaReference

#endif

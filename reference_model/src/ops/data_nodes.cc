
// Copyright (c) 2020-2022, ARM Limited.
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

#include "data_nodes.h"

using namespace TosaReference;
using namespace Eigen;
using namespace tosa;

OpConst::OpConst(SubgraphTraverser* sgt_, uint64_t id_)
    : GraphNode(sgt_, Op_CONST, id_)
{
    setRequiredOperands(0, 1);
}

OpConst::~OpConst()
{}

int OpConst::checkTensorAttributes()
{
    if (validateRequiredOperands())
        return 1;

    return 0;
}

int OpConst::eval()
{
    // Evaluation is trivial for constants
    return GraphNode::eval();
}

template <int Rank, DType Dtype>
OpIdentity<Rank, Dtype>::OpIdentity(SubgraphTraverser* sgt_,
                                    TosaAttributeBase* attribute_,
                                    uint64_t id_)
    : GraphNode(sgt_, Op_IDENTITY, id_)
{
    setRequiredOperands(1, 1);
    setRequiredRank(0, 6);
}

template <int Rank, DType Dtype>
OpIdentity<Rank, Dtype>::~OpIdentity()
{}

template <int Rank, DType Dtype>
int OpIdentity<Rank, Dtype>::checkTensorAttributes()
{

    if (inputs.size() != outputs.size())
    {
        printNodeValidationError("Input and output tensor list lengths are not equal");
        return 1;
    }

    in  = dynamic_cast<TosaReference::TensorTemplate<TIn>*>(inputs[0]);
    out = dynamic_cast<TosaReference::TensorTemplate<TOut>*>(outputs[0]);

    if (in->matchRankTypeShape(*out))
    {
        printNodeValidationError("Input and output tensor rank, type, or shape do not match");
        return 1;
    }

    return 0;
}

template <int Rank, DType Dtype>
int OpIdentity<Rank, Dtype>::eval()
{
    out->getTensor() = in->getTensor();

    return GraphNode::eval();
}

// template explicit instantiation
// note OpConst is not templated

DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpIdentity, FP16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpIdentity, FLOAT);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpIdentity, INT8);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpIdentity, INT16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpIdentity, INT32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpIdentity, BOOL);

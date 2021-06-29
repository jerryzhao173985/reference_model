
// Copyright (c) 2020-2021, ARM Limited.
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

#include "reduction.h"
#include "quant_util.h"

using namespace TosaReference;
using namespace Eigen;
using namespace tosa;

template <int Rank, DType Dtype>
ReduceNode<Rank, Dtype>::ReduceNode(SubgraphTraverser* sgt_, const Op& op_, TosaAttributeBase* attribute_, uint64_t id_)
    : GraphNode(sgt_, op_, id_)
{
    setRequiredOperands(1, 1);
    setRequiredRank(0, 4);

    INIT_ATTRIBUTE(Axis);
}

template <int Rank, DType Dtype>
ReduceNode<Rank, Dtype>::~ReduceNode()
{
    if (attribute)
        delete attribute;
}

template <int Rank, DType Dtype>
int ReduceNode<Rank, Dtype>::checkTensorAttributes()
{
    if (validateRequiredOperands())
        return 1;

    if (validateRequiredRank(inputs[0]) || validateRequiredRank(outputs[0]))
    {
        return 1;
    }

    if (attribute->axis() < 0 || attribute->axis() >= inputs[0]->getRank())
    {
        printNodeValidationError("Reduce axis must between [0, input_rank - 1]");
        return 1;
    }

    if (inputs[0]->matchRank(*outputs[0]))
    {
        printNodeValidationError("Input and output tensor ranks must match");
        return 1;
    }

    in  = dynamic_cast<TosaReference::TensorTemplate<TIn>*>(inputs[0]);
    out = dynamic_cast<TosaReference::TensorTemplate<TOut>*>(outputs[0]);

    ASSERT_MEM(in && out);

    dims[0] = this->attribute->axis();

    return 0;
}

template <int Rank, DType Dtype>
int OpReduceAll<Rank, Dtype>::eval()
{
    this->out->getTensor() = this->in->getTensor().all(this->dims).reshape(this->out->getTensor().dimensions());

    return GraphNode::eval();
}

template <int Rank, DType Dtype>
int OpReduceAny<Rank, Dtype>::eval()
{
    this->out->getTensor() = this->in->getTensor().any(this->dims).reshape(this->out->getTensor().dimensions());

    return GraphNode::eval();
}

template <int Rank, DType Dtype>
int OpReduceMax<Rank, Dtype>::eval()
{
    this->out->getTensor() = this->in->getTensor().maximum(this->dims).reshape(this->out->getTensor().dimensions());

    return GraphNode::eval();
}

template <int Rank, DType Dtype>
int OpReduceMin<Rank, Dtype>::eval()
{
    this->out->getTensor() = this->in->getTensor().minimum(this->dims).reshape(this->out->getTensor().dimensions());

    return GraphNode::eval();
}

template <int Rank, DType Dtype>
int OpReduceProduct<Rank, Dtype>::eval()
{
    this->out->getTensor() = this->in->getTensor().prod(this->dims).reshape(this->out->getTensor().dimensions());

    return GraphNode::eval();
}

template <int Rank, DType Dtype>
int OpReduceSum<Rank, Dtype>::eval()
{
    this->out->getTensor() = this->in->getTensor().sum(this->dims).reshape(this->out->getTensor().dimensions());

    return GraphNode::eval();
}

// template explicit instantiation
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceAll, BOOL);

DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceAny, BOOL);

DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceMax, FLOAT);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceMax, INT8);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceMax, INT16);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceMax, INT32);

DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceMin, FLOAT);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceMin, INT8);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceMin, INT16);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceMin, INT32);

DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceProduct, FLOAT);

DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceSum, FLOAT);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceSum, INT32);


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

#include "ewise_ternary.h"

using namespace TosaReference;
using namespace Eigen;
using namespace tosa;

template <int Rank, DType Dtype>
OpSelectBase<Rank, Dtype>::OpSelectBase(TosaAttributeBase* attribute_, TosaQuantInfoBase* qinfo_, uint64_t id_)
    : GraphNode(Op_SELECT, id_)
{
    setRequiredOperands(3, 1);
    setRequiredRank(0, 6);
}

template <int Rank, DType Dtype>
OpSelectBase<Rank, Dtype>::~OpSelectBase()
{}

template <int Rank, DType Dtype>
int OpSelectBase<Rank, Dtype>::checkTensorAttributes()
{
    if (validateRequiredOperands())
        return 1;

    if (validateRequiredRank(inputs[0]) || validateRequiredRank(inputs[1]) || validateRequiredRank(inputs[2]) ||
        validateRequiredRank(outputs[0]))
    {
        return 1;
    }

    // output and input must be the same types
    if (inputs[0]->matchRank(*outputs[0]) || inputs[1]->matchRankType(*outputs[0]) ||
        inputs[2]->matchRankType(*outputs[0]))
    {
        printNodeValidationError("Failure to match input and output rank and type");
        return 1;
    }

    cond     = dynamic_cast<TosaReference::TensorTemplate<TCond>*>(inputs[0]);
    then_val = dynamic_cast<TosaReference::TensorTemplate<TIn>*>(inputs[1]);
    else_val = dynamic_cast<TosaReference::TensorTemplate<TIn>*>(inputs[2]);
    out      = dynamic_cast<TosaReference::TensorTemplate<TIn>*>(outputs[0]);

    return 0;
}

template <int Rank, DType Dtype>
int OpSelectBase<Rank, Dtype>::eval()
{
    FATAL_ERROR_NODE("shouldn't be called");
}

template <int Rank, DType Dtype>
int OpSelect<Rank, Dtype>::broadcast()
{
    std::vector<int> cond_shape = this->cond->getShape();
    std::vector<int> then_shape = this->then_val->getShape();
    std::vector<int> else_shape = this->else_val->getShape();
    std::vector<int> out_shape  = this->out->getShape();

    for (int i = 0; i < Rank; i++)
    {
        this->bcast_cond[i] = (cond_shape[i] == 1) ? std::max(then_shape[i], else_shape[i]) : 1;
        this->bcast_then[i] = (then_shape[i] == 1) ? std::max(cond_shape[i], else_shape[i]) : 1;
        this->bcast_else[i] = (else_shape[i] == 1) ? std::max(then_shape[i], cond_shape[i]) : 1;
        ASSERT_MSG_NODE((this->bcast_cond[i] * cond_shape[i]) == out_shape[i], "SELECT broadcast invariant failed");
        ASSERT_MSG_NODE((this->bcast_then[i] * then_shape[i]) == out_shape[i], "SELECT broadcast invariant failed");
        ASSERT_MSG_NODE((this->bcast_else[i] * else_shape[i]) == out_shape[i], "SELECT broadcast invariant failed");
    }

    return 0;
}

template <int Rank, DType Dtype>
int OpSelect<Rank, Dtype>::eval()
{
    this->broadcast();
    this->out->getTensor() = this->cond->getTensor()
                                 .broadcast(this->bcast_cond)
                                 .select(this->then_val->getTensor().broadcast(this->bcast_then),
                                         this->else_val->getTensor().broadcast(this->bcast_else));

    return GraphNode::eval();
}

template <DType Dtype>
int OpSelect<0, Dtype>::eval()
{
    this->out->getTensor() = this->cond->getTensor().select(this->then_val->getTensor(), this->else_val->getTensor());

    return GraphNode::eval();
}

// template explicit instantiation
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpSelect, FLOAT);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpSelect, AINT8);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpSelect, INT8);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpSelect, INT16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpSelect, INT32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpSelect, BOOL);

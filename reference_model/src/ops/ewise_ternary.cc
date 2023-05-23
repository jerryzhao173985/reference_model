
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

#include "ewise_ternary.h"

using namespace TosaReference;
using namespace Eigen;
using namespace tosa;

template <int Rank, TOSA_REF_TYPE Dtype>
OpSelectBase<Rank, Dtype>::OpSelectBase(SubgraphTraverser* sgt_,
                                        TosaAttributeBase* attribute_,
                                        uint64_t id_)
    : GraphNode(sgt_, Op_SELECT, id_)
{
    setRequiredOperands(3, 1);
}

template <int Rank, TOSA_REF_TYPE Dtype>
OpSelectBase<Rank, Dtype>::~OpSelectBase()
{}

template <int Rank, TOSA_REF_TYPE Dtype>
int OpSelectBase<Rank, Dtype>::checkTensorAttributes()
{
    // Check Tosa Level
    auto tosa_level = g_func_config.tosa_level;
    LEVEL_CHECK(Rank <= tosa_level.MAX_RANK, "Rank should be smaller than or equal to MAX_RANK");

    if (validateRequiredOperands())
        return 1;

    // output and input must be the same types
    if (inputs[0]->matchRankShape(*outputs[0], true /* broadcastOk */) ||
        inputs[1]->matchRankTypeShape(*outputs[0], true /* broadcastOk */) ||
        inputs[2]->matchRankTypeShape(*outputs[0], true /* broadcastOk */))
    {
        printNodeValidationError("Failure to match input and output rank/type/shape");
        return 1;
    }

    cond     = dynamic_cast<TosaReference::TensorTemplate<TCond>*>(inputs[0]);
    then_val = dynamic_cast<TosaReference::TensorTemplate<TIn>*>(inputs[1]);
    else_val = dynamic_cast<TosaReference::TensorTemplate<TIn>*>(inputs[2]);
    out      = dynamic_cast<TosaReference::TensorTemplate<TIn>*>(outputs[0]);

    return 0;
}

template <int Rank, TOSA_REF_TYPE Dtype>
int OpSelectBase<Rank, Dtype>::eval()
{
    FATAL_ERROR("shouldn't be called");
}

template <int Rank, TOSA_REF_TYPE Dtype>
int OpSelect<Rank, Dtype>::broadcast(std::vector<int>& calculated_shape)
{
    const std::vector<int>& cond_shape   = this->cond->getShape();
    const std::vector<int>& then_shape   = this->then_val->getShape();
    const std::vector<int>& else_shape   = this->else_val->getShape();
    const std::vector<int>& output_shape = this->out->getShape();

    // calculates the multipliers for Eigen
    for (int i = 0; i < Rank; i++)
    {
        this->bcast_cond[i] = (cond_shape[i] != output_shape[i] && cond_shape[i] == 1) ? output_shape[i] : 1;
        this->bcast_then[i] = (then_shape[i] != output_shape[i] && then_shape[i] == 1) ? output_shape[i] : 1;
        this->bcast_else[i] = (else_shape[i] != output_shape[i] && else_shape[i] == 1) ? output_shape[i] : 1;
    }

    // calculates the broadcasted output shape
    calculated_shape = cond_shape;
    for (size_t i = 0; i < calculated_shape.size(); i++) {
        if (calculated_shape[i] == 1) {
            calculated_shape[i] = then_shape[i];
        } else {
            ERROR_IF(then_shape[i] != 1 && then_shape[i] != calculated_shape[i], "Broadcast_shape failure, input shapes are not compatible");
        }

        if (calculated_shape[i] == 1) {
            calculated_shape[i] = else_shape[i];
        } else {
            ERROR_IF(else_shape[i] != 1 && else_shape[i] != calculated_shape[i], "Broadcast_shape failure, input shapes are not compatible");
        }
    }

    return 0;
}

template <int Rank, TOSA_REF_TYPE Dtype>
int OpSelect<Rank, Dtype>::eval()
{
    std::vector<int> calculated_shape;
    this->broadcast(calculated_shape);

    auto result_shape = this->out->getShape();
    ERROR_IF(calculated_shape != result_shape, "Broadcast_shape failure, calculated_shape and result_shape don't match");

    this->out->getTensor() = this->cond->getTensor()
                                 .broadcast(this->bcast_cond)
                                 .select(this->then_val->getTensor().broadcast(this->bcast_then),
                                         this->else_val->getTensor().broadcast(this->bcast_else));

    return GraphNode::eval();
}

template <TOSA_REF_TYPE Dtype>
int OpSelect<0, Dtype>::eval()
{
    this->out->getTensor() = this->cond->getTensor().select(this->then_val->getTensor(), this->else_val->getTensor());

    return GraphNode::eval();
}

// template explicit instantiation
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpSelectBase, FP16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpSelectBase, BF16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpSelectBase, FP32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpSelectBase, INT8);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpSelectBase, INT16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpSelectBase, INT32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpSelectBase, BOOL);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpSelectBase, FP64);

DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpSelect, FP16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpSelect, BF16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpSelect, FP32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpSelect, INT8);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpSelect, INT16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpSelect, INT32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpSelect, BOOL);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpSelect, FP64);

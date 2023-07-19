
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

#include "scatter_gather.h"
#include "quant_util.h"

using namespace TosaReference;
using namespace Eigen;
using namespace tosa;

template <TOSA_REF_TYPE Dtype>
OpGather<Dtype>::OpGather(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_)
    : GraphNode(sgt_, Op_GATHER, id_)
{
    setRequiredOperands(2, 1);
}

template <TOSA_REF_TYPE Dtype>
OpGather<Dtype>::~OpGather()
{}

template <TOSA_REF_TYPE Dtype>
int OpGather<Dtype>::checkTensorAttributes()
{
    if (validateRequiredOperands())
        return 1;

    if (inputs[0]->getRank() != 3)
    {
        printNodeValidationError("OpGather: values needs to be rank 3 tensor");
        return 1;
    }

    if (inputs[1]->getRank() != 2)
    {
        printNodeValidationError("OpGather: indices needs to be rank 2 tensor");
        return 1;
    }

    if (outputs[0]->getRank() != 3)
    {
        printNodeValidationError("OpGather: output needs to be rank 3 tensor");
        return 1;
    }

    K = inputs[0]->getShape()[1];
    N = outputs[0]->getShape()[0];
    W = outputs[0]->getShape()[1];
    C = outputs[0]->getShape()[2];

    if (N != inputs[0]->getShape()[0] || N != inputs[1]->getShape()[0])
    {
        printNodeValidationError("OpGather: dimension N mismatch");
        return 1;
    }

    if (W != inputs[1]->getShape()[1])
    {
        printNodeValidationError("OpGather: dimension W mismatch");
        return 1;
    }

    if (C != inputs[0]->getShape()[2])
    {
        printNodeValidationError("OpGather: dimension C mismatch");
        return 1;
    }

    // output and input must be the same types
    if (inputs[0]->matchType(*outputs[0]))
    {
        printNodeValidationError("Failure to match input and output type");
        return 1;
    }

    values  = dynamic_cast<TosaReference::TensorTemplate<TValue>*>(inputs[0]);
    indices = dynamic_cast<TosaReference::TensorTemplate<TIndex>*>(inputs[1]);
    output  = dynamic_cast<TosaReference::TensorTemplate<TOutput>*>(outputs[0]);

    ASSERT_MEM(values && indices && output);

    return 0;
}

template <TOSA_REF_TYPE Dtype>
int OpGather<Dtype>::eval()
{
    for (int32_t n = 0; n < N; n++)
    {
        for (int32_t w = 0; w < W; w++)
        {
            int32_t k = this->indices->getTensor()(n, w);
            REQUIRE(k >= 0 && k < K, "OpGather: index(%d, %d)=%d exceed valid range [0, %d]", n, w, k, K);
            for (int32_t c = 0; c < C; c++)
            {
                EigenType value                    = this->values->getTensor()(n, k, c);
                this->output->getTensor()(n, w, c) = value;
            }
        }
    }

    return GraphNode::eval();
}

template <TOSA_REF_TYPE Dtype>
OpScatter<Dtype>::OpScatter(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_)
    : GraphNode(sgt_, Op_SCATTER, id_)
{
    setRequiredOperands(3, 1);
}

template <TOSA_REF_TYPE Dtype>
OpScatter<Dtype>::~OpScatter()
{}

template <TOSA_REF_TYPE Dtype>
int OpScatter<Dtype>::checkTensorAttributes()
{
    if (validateRequiredOperands())
        return 1;

    if (inputs[0]->getRank() != 3)
    {
        printNodeValidationError("OpGather: values_in needs to be rank 3 tensor");
        return 1;
    }

    if (inputs[1]->getRank() != 2)
    {
        printNodeValidationError("OpGather: indices needs to be rank 2 tensor");
        return 1;
    }

    if (inputs[2]->getRank() != 3)
    {
        printNodeValidationError("OpGather: input needs to be rank 3 tensor");
        return 1;
    }

    if (outputs[0]->getRank() != 3)
    {
        printNodeValidationError("OpGather: values_out needs to be rank 3 tensor");
        return 1;
    }

    W = inputs[2]->getShape()[1];
    N = outputs[0]->getShape()[0];
    K = outputs[0]->getShape()[1];
    C = outputs[0]->getShape()[2];

    if (N != inputs[0]->getShape()[0] || N != inputs[1]->getShape()[0] || N != inputs[2]->getShape()[0])
    {
        printNodeValidationError("OpScatter: dimension N mismatch");
        return 1;
    }

    if (W != inputs[1]->getShape()[1])
    {
        printNodeValidationError("OpGather: dimension W mismatch");
        return 1;
    }

    if (C != inputs[0]->getShape()[2] || C != inputs[2]->getShape()[2])
    {
        printNodeValidationError("OpGather: dimension C mismatch");
        return 1;
    }

    // output and input must be the same types
    if (inputs[0]->matchType(*outputs[0]))
    {
        printNodeValidationError("Failure to match input and output type");
        return 1;
    }

    values_in  = dynamic_cast<TosaReference::TensorTemplate<TValue>*>(inputs[0]);
    indices    = dynamic_cast<TosaReference::TensorTemplate<TIndex>*>(inputs[1]);
    input      = dynamic_cast<TosaReference::TensorTemplate<TValue>*>(inputs[2]);
    values_out = dynamic_cast<TosaReference::TensorTemplate<TOutput>*>(outputs[0]);

    ASSERT_MEM(values_in && indices && input && values_out);

    return 0;
}

template <TOSA_REF_TYPE Dtype>
int OpScatter<Dtype>::eval()
{
    // Initializes the output tensor with the input value for values that are unchanged by the scatter operation.
    this->values_out->getTensor() = this->values_in->getTensor();

    for (int n = 0; n < N; n++)
    {
        for (int w = 0; w < W; w++)
        {
            int32_t k = this->indices->getTensor()(n, w);
            REQUIRE(k >= 0 && k < K, "OpScatter: index(%d, %d)=%d exceed valid range [0, %d]", n, w, k, K);
            for (int c = 0; c < C; c++)
            {
                EigenType value                        = this->input->getTensor()(n, w, c);
                this->values_out->getTensor()(n, k, c) = value;
            }
        }
    }

    return GraphNode::eval();
}

// template explicit instantiation
DEF_INSTANTIATE_ONE_TYPE(OpGather, INT8);
DEF_INSTANTIATE_ONE_TYPE(OpGather, INT16);
DEF_INSTANTIATE_ONE_TYPE(OpGather, INT32);
DEF_INSTANTIATE_ONE_TYPE(OpGather, FP16);
DEF_INSTANTIATE_ONE_TYPE(OpGather, BF16);
DEF_INSTANTIATE_ONE_TYPE(OpGather, FP32);
DEF_INSTANTIATE_ONE_TYPE(OpGather, FP64);

DEF_INSTANTIATE_ONE_TYPE(OpScatter, INT8);
DEF_INSTANTIATE_ONE_TYPE(OpScatter, INT16);
DEF_INSTANTIATE_ONE_TYPE(OpScatter, INT32);
DEF_INSTANTIATE_ONE_TYPE(OpScatter, FP16);
DEF_INSTANTIATE_ONE_TYPE(OpScatter, BF16);
DEF_INSTANTIATE_ONE_TYPE(OpScatter, FP32);
DEF_INSTANTIATE_ONE_TYPE(OpScatter, FP64);

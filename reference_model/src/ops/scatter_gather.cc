
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

#include "scatter_gather.h"
#include "quant_util.h"

using namespace TosaReference;
using namespace Eigen;
using namespace tosa;

template <int InRank, int IndexRank, DType Dtype>
OpGather<InRank, IndexRank, Dtype>::OpGather(TosaAttributeBase* attribute_, TosaQuantInfoBase* qinfo_, uint64_t id_)
    : GraphNode(Op_GATHER, id_)
{
    setRequiredOperands(2, 1);
    setRequiredRank(1, 6);

    INIT_ATTRIBUTE(Axis);
}

template <int InRank, int IndexRank, DType Dtype>
OpGather<InRank, IndexRank, Dtype>::~OpGather()
{
    if (attribute)
        delete attribute;
}

template <int InRank, int IndexRank, DType Dtype>
int OpGather<InRank, IndexRank, Dtype>::checkTensorAttributes()
{
    if (validateRequiredOperands())
        return 1;

    if (validateRequiredRank(inputs[0]) || validateRequiredRank(inputs[1]) || validateRequiredRank(outputs[0]))
    {
        return 1;
    }

    // output and input must be the same types
    if (inputs[0]->matchType(*outputs[0]))
    {
        printNodeValidationError("Failure to match input and output type");
        return 1;
    }

    in    = dynamic_cast<TosaReference::TensorTemplate<TIn>*>(inputs[0]);
    index = dynamic_cast<TosaReference::TensorTemplate<TIndex>*>(inputs[1]);
    out   = dynamic_cast<TosaReference::TensorTemplate<TOut>*>(outputs[0]);

    ASSERT_MEM(in && index && out);

    return 0;
}

template <int InRank, int IndexRank, DType Dtype>
int OpGather<InRank, IndexRank, Dtype>::eval()
{
    int axis = attribute->axis();

    // calculate size left and right to axis
    int left_size = 1;
    for (int i = 0; i < axis; ++i)
    {
        left_size *= in->getShape()[i];
    }

    int right_size = 1;
    for (int i = axis + 1; i < in->getRank(); ++i)
    {
        right_size *= in->getShape()[i];
    }

    InEigenType* input_data   = in->getTensor().data();
    int32_t* index_data       = index->getTensor().data();
    OutEigenType* output_data = out->getTensor().data();

    int32_t axis_size   = in->getShape()[axis];
    int32_t index_count = index->getElementCount();

    // sanity check if index is valid
    // need to check until this point since index is not known until runtime
    for (size_t i = 0; i < index->getElementCount(); i++)
    {
        if (index_data[i] >= axis_size)
        {
            FATAL_ERROR_NODE("OpGather: index[%lu]=%i can't exceed axis_size=%i", i, index_data[i], axis_size);
        }
    }

    // Eigen stores tensor in column-major
    // so we iterate through dimension right to axis and the index array
    // do memory copy with size of left size each time
    for (int right = 0; right < right_size; ++right)
    {
        for (int i = 0; i < index_count; ++i)
        {
            std::memcpy(output_data + (right * index_count + i) * left_size,
                        input_data + (right * axis_size + index_data[i]) * left_size, sizeof(InEigenType) * left_size);
        }
    }

    return GraphNode::eval();
}

// template explicit instantiation
DEF_INSTANTIATE_GATHER(OpGather, AINT8);
DEF_INSTANTIATE_GATHER(OpGather, INT16);
DEF_INSTANTIATE_GATHER(OpGather, INT32);

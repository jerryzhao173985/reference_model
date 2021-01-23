
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

#include "data_layout.h"
#include "quant_util.h"

using namespace TosaReference;
using namespace Eigen;
using namespace tosa;

template <int Rank, DType Dtype>
OpConcat<Rank, Dtype>::OpConcat(TosaAttributeBase* attribute_, TosaQuantInfoBase* qinfo_, uint64_t id_)
    : GraphNode(Op_CONCAT, id_)
{
    setRequiredOperands(2, 1);
    setRequiredRank(1, 6);

    INIT_ATTRIBUTE(Axis);
}

template <int Rank, DType Dtype>
OpConcat<Rank, Dtype>::~OpConcat()
{
    if (attribute)
        delete attribute;
}

template <int Rank, DType Dtype>
int OpConcat<Rank, Dtype>::checkTensorAttributes()
{
    if (validateRequiredOperands())
        return 1;

    if (validateRequiredRank(inputs[0]) || validateRequiredRank(outputs[0]))
    {
        return 1;
    }

    // output and input must be the same types and rank
    // inputs[0] and inputs[1] should also match type and rank
    if (inputs[0]->matchRankType(*outputs[0]) || inputs[1]->matchRankType(*outputs[0]))
    {
        printNodeValidationError("Concat operator input ranks and types must match");
        return 1;
    }

    lhs = dynamic_cast<TosaReference::TensorTemplate<TIn>*>(inputs[0]);
    rhs = dynamic_cast<TosaReference::TensorTemplate<TIn>*>(inputs[1]);
    out = dynamic_cast<TosaReference::TensorTemplate<TOut>*>(outputs[0]);

    if (attribute->axis() < 0 || (size_t)attribute->axis() >= rhs->getShape().size())
    {
        printNodeValidationError("Axis is beyond input tensor rank");
        return 1;
    }

    return 0;
}

template <int Rank, DType Dtype>
int OpConcat<Rank, Dtype>::eval()
{

    int32_t reversed_axis = Rank - 1 - attribute->axis();

    for (int32_t d = 0; d < Rank; d++)
    {
        reverser[d] = Rank - 1 - d;
    }

    TIn lhs_reversed = lhs->getTensor().shuffle(reverser);
    TIn rhs_reversed = rhs->getTensor().shuffle(reverser);

    TIn reversed_result = lhs_reversed.concatenate(rhs_reversed, reversed_axis);
    out->getTensor()    = reversed_result.shuffle(reverser);
    //    out->getTensor() = lhs->getTensor().concatenate(rhs->getTensor(), axis);

    return GraphNode::eval();
}

template <int Rank, DType Dtype>
OpPad<Rank, Dtype>::OpPad(TosaAttributeBase* attribute_, TosaQuantInfoBase* qinfo_, uint64_t id_)
    : GraphNode(Op_PAD, id_)
{
    setRequiredOperands(2, 1);
    setRequiredRank(0, 6);

    INIT_QINFO(Pad);
}

template <int Rank, DType Dtype>
OpPad<Rank, Dtype>::~OpPad()
{
    if (qinfo)
        delete qinfo;
}

template <int Rank, DType Dtype>
int OpPad<Rank, Dtype>::checkTensorAttributes()
{
    if (validateRequiredOperands())
        return 1;

    if (validateRequiredRank(inputs[0]) || validateRequiredRank(outputs[0]))
    {
        return 1;
    }

    // output and input must be the same types
    if (inputs[0]->matchRankType(*outputs[0]))
    {
        printNodeValidationError("Failure to match input and output type and rank");
        return 1;
    }

    in  = dynamic_cast<TosaReference::TensorTemplate<TIn>*>(inputs[0]);
    out = dynamic_cast<TosaReference::TensorTemplate<TOut>*>(outputs[0]);
    TosaReference::TensorTemplate<ETensor2<int32_t>>* paddings =
        dynamic_cast<TosaReference::TensorTemplate<ETensor2<int32_t>>*>(inputs[1]);

    for (int i = 0; i < Rank; i++)
    {
        paddings_array[i] = std::make_pair(paddings->getTensor()(i, 0), paddings->getTensor()(i, 1));
    }

    return 0;
}

template <int Rank, DType Dtype>
int OpPad<Rank, Dtype>::eval()
{
    InEigenType pad_value = 0;
    if (this->qinfo)
    {
        pad_value = (InEigenType)this->qinfo->input_zp();
    }

    this->out->getTensor() = this->in->getTensor().pad(this->paddings_array, pad_value);

    return GraphNode::eval();
}

template <int InRank, int OutRank, DType Dtype>
OpReshape<InRank, OutRank, Dtype>::OpReshape(TosaAttributeBase* attribute_, TosaQuantInfoBase* qinfo_, uint64_t id_)
    : GraphNode(Op_RESHAPE, id_)
{
    setRequiredOperands(1, 1);
    setRequiredRank(0, 6);

    INIT_ATTRIBUTE(Reshape);
}

template <int InRank, int OutRank, DType Dtype>
OpReshape<InRank, OutRank, Dtype>::~OpReshape()
{
    if (attribute)
        delete attribute;
}

template <int InRank, int OutRank, DType Dtype>
int OpReshape<InRank, OutRank, Dtype>::checkTensorAttributes()
{
    uint32_t minusOneCount = 0;

    if (validateRequiredOperands())
        return 1;

    if (validateRequiredRank(inputs[0]) || validateRequiredRank(outputs[0]))
    {
        return 1;
    }

    // output and input must be the same types
    if (inputs[0]->matchType(*outputs[0]))
    {
        printNodeValidationError("OpReshape: Input and output types must match");
        return 1;
    }

    for (uint32_t d = 0; d < OutRank; d++)
    {
        if (attribute->shape()[d] == -1)
        {
            minusOneCount++;
        }
    }

    if (minusOneCount > 1)
    {
        printNodeValidationError("OpReshape: new shape has more than one -1 dimension");
        return 1;
    }

    in  = dynamic_cast<TosaReference::TensorTemplate<TIn>*>(inputs[0]);
    out = dynamic_cast<TosaReference::TensorTemplate<TOut>*>(outputs[0]);

    return 0;
}

template <int InRank, int OutRank, DType Dtype>
int OpReshape<InRank, OutRank, Dtype>::eval()
{
    uint32_t remainingSize = in->getElementCount();

    // If there is a -1 dimension, find the remainder in one pass over the output shape
    for (int32_t d = 0; d < OutRank; d++)
    {
        if (attribute->shape()[d] != -1)
        {
            remainingSize = remainingSize / attribute->shape()[d];
        }
    }

    for (int32_t d = 0; d < OutRank; d++)
    {
        array_shape[d]  = attribute->shape()[OutRank - 1 - d];
        out_reverser[d] = OutRank - 1 - d;

        // Jam in the remainder here
        if (array_shape[d] == -1)
        {
            array_shape[d] = remainingSize;
        }
    }

    for (int32_t d = 0; d < InRank; d++)
    {
        in_reverser[d] = InRank - 1 - d;
    }

    // Eigen Tensor is col-major, and we're referencing row-major result
    // need to reverse it to row-major before reshape, and perform another reverse afterward

    // input tensor rank 0 can't do .shuffle(), need to be handled otherwise
    TIn in_reversed;
    if (InRank > 1)
    {
        in_reversed = in->getTensor().shuffle(in_reverser);
    }
    else
    {
        in_reversed = in->getTensor();
    }

    TOut in_reshaped = in_reversed.reshape(array_shape);

    // output tensor can be rank 0, .reshape() and .shuffle() don't work, need to be handled otherwise
    if (OutRank > 1)
    {
        out->getTensor() = in_reshaped.shuffle(out_reverser);
    }
    else
    {
        out->getTensor() = in_reshaped;
    }

    return GraphNode::eval();
}

template <int Rank, DType Dtype>
OpReverse<Rank, Dtype>::OpReverse(TosaAttributeBase* attribute_, TosaQuantInfoBase* qinfo_, uint64_t id_)
    : GraphNode(Op_REVERSE, id_)
{
    setRequiredOperands(1, 1);
    setRequiredRank(1, 6);

    INIT_ATTRIBUTE(Axis);
}

template <int Rank, DType Dtype>
OpReverse<Rank, Dtype>::~OpReverse()
{
    if (attribute)
        delete attribute;
}

template <int Rank, DType Dtype>
int OpReverse<Rank, Dtype>::checkTensorAttributes()
{
    if (validateRequiredOperands())
        return 1;

    if (validateRequiredRank(inputs[0]) || validateRequiredRank(outputs[0]))
    {
        return 1;
    }

    // output and input must be the same types
    if (inputs[0]->matchRankTypeShape(*outputs[0]))
    {
        printNodeValidationError("Failure to match input and output rank/type/shape");
        return 1;
    }

    in  = dynamic_cast<TosaReference::TensorTemplate<TIn>*>(inputs[0]);
    out = dynamic_cast<TosaReference::TensorTemplate<TOut>*>(outputs[0]);

    ASSERT_MEM(in && out);

    if (attribute->axis() < 0 || attribute->axis() >= inputs[0]->getRank())
    {
        printNodeValidationError("Reverse axis must between [0, input_rank - 1]");
        return 1;
    }

    // transform list of axis into true or false list
    // e.g. rank=4, axis=[1,2], reverse array would be [false, true, true, false]
    for (int i = 0; i < Rank; i++)
    {
        reverse_array[i] = false;
    }
    reverse_array[attribute->axis()] = true;

    return 0;
}

template <int Rank, DType Dtype>
int OpReverse<Rank, Dtype>::eval()
{
    out->getTensor() = in->getTensor().reverse(reverse_array);

    return GraphNode::eval();
}

template <int Rank, DType Dtype>
OpSlice<Rank, Dtype>::OpSlice(TosaAttributeBase* attribute_, TosaQuantInfoBase* qinfo_, uint64_t id_)
    : GraphNode(Op_SLICE, id_)
{
    setRequiredOperands(1, 1);
    setRequiredRank(0, 6);

    INIT_ATTRIBUTE(Slice);
}

template <int Rank, DType Dtype>
OpSlice<Rank, Dtype>::~OpSlice()
{
    if (attribute)
        delete attribute;
}

template <int Rank, DType Dtype>
int OpSlice<Rank, Dtype>::checkTensorAttributes()
{
    if (validateRequiredOperands())
        return 1;

    if (validateRequiredRank(inputs[0]) || validateRequiredRank(outputs[0]))
    {
        return 1;
    }

    // output and input must be the same types
    if (inputs[0]->matchType(*outputs[0]))
    {
        printNodeValidationError("Failure to match input and output type");
        return 1;
    }

    in  = dynamic_cast<TosaReference::TensorTemplate<TIn>*>(inputs[0]);
    out = dynamic_cast<TosaReference::TensorTemplate<TOut>*>(outputs[0]);

    for (size_t i = 0; i < attribute->begin().size(); i++)
    {
        begin_array[i] = attribute->begin()[i];
    }

    for (size_t i = 0; i < attribute->size().size(); i++)
    {
        if (attribute->size()[i] != 0)
        {
            size_array[i] = attribute->size()[i];
        }
        else
        {
            // Tensorflow assigns a zero size to dimensions that are kept
            // Eigen expects size to be the full size of the dimension
            size_array[i] = in->getTensor().dimension(0);
        }
    }

    return 0;
}

template <int Rank, DType Dtype>
int OpSlice<Rank, Dtype>::eval()
{
    out->getTensor() = in->getTensor().slice(begin_array, size_array);

    return GraphNode::eval();
}

template <int Rank, DType Dtype>
OpTileBase<Rank, Dtype>::OpTileBase(TosaAttributeBase* attribute_, TosaQuantInfoBase* qinfo_, uint64_t id_)
    : GraphNode(Op_TILE, id_)
{
    setRequiredOperands(1, 1);
    setRequiredRank(0, 6);

    INIT_ATTRIBUTE(Tile);
}

template <int Rank, DType Dtype>
OpTileBase<Rank, Dtype>::~OpTileBase()
{
    if (attribute)
        delete attribute;
}

template <int Rank, DType Dtype>
int OpTileBase<Rank, Dtype>::checkTensorAttributes()
{
    if (validateRequiredOperands())
        return 1;

    if (validateRequiredRank(inputs[0]) || validateRequiredRank(outputs[0]))
    {
        return 1;
    }

    // output and input must be the same ranks and types
    if (inputs[0]->matchRankType(*outputs[0]))
    {
        printNodeValidationError("Failure to match input and output rank or type");
        return 1;
    }

    in  = dynamic_cast<TosaReference::TensorTemplate<TIn>*>(inputs[0]);
    out = dynamic_cast<TosaReference::TensorTemplate<TOut>*>(outputs[0]);

    if (attribute->multiples().size() != Rank)
    {
        printNodeValidationError("1D list 'multiples' must have size equal to input rank");
        return 1;
    }

    for (int32_t d = 0; d < Rank; d++)
    {
        if (in->getShape()[d] * attribute->multiples()[d] != out->getShape()[d])
        {
            printNodeValidationError("unexpected output shape");
            return 1;
        }
    }

    return 0;
}

template <int Rank, DType Dtype>
int OpTile<Rank, Dtype>::eval()
{
    // primary template shouldn't be called
    FATAL_ERROR_NODE("OpTile rank=%i, dtype=%s: not implemented yet", Rank, EnumNamesDType()[Dtype]);
}

template <DType Dtype>
int OpTile<1, Dtype>::eval()
{
    for (int32_t od0 = 0; od0 < this->out->getShape()[0]; od0++)
    {
        int32_t id0                 = od0 % this->in->getShape()[0];
        this->out->getTensor()(od0) = this->in->getTensor()(id0);
    }

    return GraphNode::eval();
}

template <DType Dtype>
int OpTile<2, Dtype>::eval()
{
    for (int32_t od0 = 0; od0 < this->out->getShape()[0]; od0++)
    {
        int32_t id0 = od0 % this->in->getShape()[0];
        for (int32_t od1 = 0; od1 < this->out->getShape()[1]; od1++)
        {
            int32_t id1                      = od1 % this->in->getShape()[1];
            this->out->getTensor()(od0, od1) = this->in->getTensor()(id0, id1);
        }
    }

    return GraphNode::eval();
}

template <DType Dtype>
int OpTile<3, Dtype>::eval()
{
    for (int32_t od0 = 0; od0 < this->out->getShape()[0]; od0++)
    {
        int32_t id0 = od0 % this->in->getShape()[0];
        for (int32_t od1 = 0; od1 < this->out->getShape()[1]; od1++)
        {
            int32_t id1 = od1 % this->in->getShape()[1];
            for (int32_t od2 = 0; od2 < this->out->getShape()[2]; od2++)
            {
                int32_t id2                           = od2 % this->in->getShape()[2];
                this->out->getTensor()(od0, od1, od2) = this->in->getTensor()(id0, id1, id2);
            }
        }
    }

    return GraphNode::eval();
}

template <DType Dtype>
int OpTile<4, Dtype>::eval()
{
    for (int32_t od0 = 0; od0 < this->out->getShape()[0]; od0++)
    {
        int32_t id0 = od0 % this->in->getShape()[0];
        for (int32_t od1 = 0; od1 < this->out->getShape()[1]; od1++)
        {
            int32_t id1 = od1 % this->in->getShape()[1];
            for (int32_t od2 = 0; od2 < this->out->getShape()[2]; od2++)
            {
                int32_t id2 = od2 % this->in->getShape()[2];
                for (int32_t od3 = 0; od3 < this->out->getShape()[3]; od3++)
                {
                    int32_t id3                                = od3 % this->in->getShape()[3];
                    this->out->getTensor()(od0, od1, od2, od3) = this->in->getTensor()(id0, id1, id2, id3);
                }
            }
        }
    }

    return GraphNode::eval();
}

template <int Rank, DType Dtype>
OpTranspose<Rank, Dtype>::OpTranspose(TosaAttributeBase* attribute_, TosaQuantInfoBase* qinfo_, uint64_t id_)
    : GraphNode(Op_TRANSPOSE, id_)
{
    setRequiredOperands(2, 1);
    setRequiredRank(0, 6);
}

template <int Rank, DType Dtype>
OpTranspose<Rank, Dtype>::~OpTranspose()
{}

template <int Rank, DType Dtype>
int OpTranspose<Rank, Dtype>::checkTensorAttributes()
{
    if (validateRequiredOperands())
        return 1;

    if (validateRequiredRank(inputs[0]) || validateRequiredRank(outputs[0]))
    {
        return 1;
    }

    // output and input must be the same types
    if (inputs[0]->matchRankType(*outputs[0]))
    {
        printNodeValidationError("Failure to match input and output rank and type");
        return 1;
    }

    if (inputs[0]->getElementCount() != outputs[0]->getElementCount())
    {
        printNodeValidationError("Failure to match input and output total element count");
        return 1;
    }

    in          = dynamic_cast<TosaReference::TensorTemplate<TIn>*>(inputs[0]);
    out         = dynamic_cast<TosaReference::TensorTemplate<TOut>*>(outputs[0]);
    perm_tensor = dynamic_cast<TosaReference::TensorTemplate<ETensor1<int32_t>>*>(inputs[1]);

    return 0;
}

template <int Rank, DType Dtype>
int OpTranspose<Rank, Dtype>::eval()
{
    for (int32_t d = 0; d < Rank; d++)
    {
        perm_array[d] = this->perm_tensor->getTensor().data()[d];
    }

    out->getTensor() = in->getTensor().shuffle(perm_array);

    return GraphNode::eval();
}

// template explicit instantiation
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpConcat, FLOAT)
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpConcat, INT8)
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpConcat, INT16)
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpConcat, INT32)
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpConcat, BOOL)

DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpPad, FLOAT);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpPad, INT8);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpPad, INT16);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpPad, INT32);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpPad, BOOL);

DEF_INSTANTIATE_RESHAPE(OpReshape, FLOAT);
DEF_INSTANTIATE_RESHAPE(OpReshape, INT8);
DEF_INSTANTIATE_RESHAPE(OpReshape, INT16);
DEF_INSTANTIATE_RESHAPE(OpReshape, INT32);
DEF_INSTANTIATE_RESHAPE(OpReshape, BOOL);

DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpReverse, FLOAT);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpReverse, INT8);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpReverse, INT16);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpReverse, INT32);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpReverse, BOOL);

DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpSlice, FLOAT);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpSlice, INT8);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpSlice, INT16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpSlice, INT32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpSlice, BOOL);

DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpTile, FLOAT);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpTile, INT8);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpTile, INT16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpTile, INT32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpTile, BOOL);

DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpTranspose, FLOAT);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpTranspose, INT8);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpTranspose, INT16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpTranspose, INT32);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpTranspose, BOOL);

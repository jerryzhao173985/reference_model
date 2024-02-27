
// Copyright (c) 2020-2024, ARM Limited.
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

template <int Rank, TOSA_REF_TYPE Dtype>
OpConcat<Rank, Dtype>::OpConcat(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_)
    : GraphNode(sgt_, Op_CONCAT, id_)
{
    setRequiredOperands(-1, 1);
    setRequiredRank(1);

    INIT_ATTRIBUTE(Axis);
}

template <int Rank, TOSA_REF_TYPE Dtype>
OpConcat<Rank, Dtype>::~OpConcat()
{
    if (attribute)
        delete attribute;
}

template <int Rank, TOSA_REF_TYPE Dtype>
int OpConcat<Rank, Dtype>::checkTensorAttributes()
{
    // Check Tosa Level
    auto tosa_level = g_func_config.tosa_level;
    LEVEL_CHECK(Rank <= tosa_level.MAX_RANK, "Rank should be smaller than or equal to MAX_RANK");

    if (validateRequiredOperands())
        return 1;

    if (inputs.empty())
    {
        printNodeValidationError("Concat operator must have at least one input tensor");
        return 1;
    }

    int32_t num_inputs = inputs.size();

    // output and input must be the same types and rank
    for (int32_t i = 0; i < num_inputs; i++)
    {
        if (inputs[i]->matchRankType(*outputs[0]))
        {
            printNodeValidationError("OpConcat: input ranks and types must match");
            return 1;
        }
        ins.push_back(dynamic_cast<TosaReference::TensorTemplate<TIn>*>(inputs[i]));
    }

    if (attribute->axis() < 0 || (size_t)attribute->axis() >= Rank)
    {
        printNodeValidationError("OpConcat: axis is beyond output tensor rank");
        return 1;
    }

    int32_t output_dim_on_axis = 0;
    for (int32_t j = 0; j < num_inputs; j++)
    {
        for (int32_t i = 0; i < Rank; i++)
        {
            int32_t input_dim = inputs[j]->getShape()[i];
            if (i == attribute->axis())
            {
                output_dim_on_axis += input_dim;
            }
            else if (input_dim != outputs[0]->getShape()[i])
            {
                printNodeValidationError("OpConcat: input dimension not matching output dimension");
                return 1;
            }
        }
    }

    ERROR_IF(output_dim_on_axis != outputs[0]->getShape()[attribute->axis()],
             "OpConcat: sum of input dimension on axis not equal to output dimension on axis");

    out = dynamic_cast<TosaReference::TensorTemplate<TOut>*>(outputs[0]);

    return 0;
}

template <int Rank, TOSA_REF_TYPE Dtype>
int OpConcat<Rank, Dtype>::eval()
{

    int32_t reversed_axis = Rank - 1 - attribute->axis();

    for (int32_t d = 0; d < Rank; d++)
    {
        reverser[d] = Rank - 1 - d;
    }

    TIn result = ins[0]->getTensor().shuffle(reverser);

    for (size_t i = 1; i < ins.size(); i++)
    {
        TIn in_reversed = ins[i]->getTensor().shuffle(reverser);
        TIn temp        = result.concatenate(in_reversed, reversed_axis);
        result          = temp;
    }
    out->getTensor() = result.shuffle(reverser);

    return GraphNode::eval();
}

template <int Rank, TOSA_REF_TYPE Dtype>
OpPad<Rank, Dtype>::OpPad(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_)
    : GraphNode(sgt_, Op_PAD, id_)
{
    setRequiredOperands(2, 1);
    setRequiredRank(1);

    INIT_ATTRIBUTE(Pad);
}

template <int Rank, TOSA_REF_TYPE Dtype>
OpPad<Rank, Dtype>::~OpPad()
{}

template <int Rank, TOSA_REF_TYPE Dtype>
int OpPad<Rank, Dtype>::checkTensorAttributes()
{
    // Check Tosa Level
    auto tosa_level = g_func_config.tosa_level;
    LEVEL_CHECK(Rank <= tosa_level.MAX_RANK, "Rank should be smaller than or equal to MAX_RANK");

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

    in      = dynamic_cast<TosaReference::TensorTemplate<TIn>*>(inputs[0]);
    padding = dynamic_cast<TosaReference::TensorTemplate<TPadding>*>(inputs[1]);
    out     = dynamic_cast<TosaReference::TensorTemplate<TOut>*>(outputs[0]);
    ASSERT_MEM(in && out);

    return 0;
}

template <int Rank, TOSA_REF_TYPE Dtype>
int OpPad<Rank, Dtype>::eval()
{
    InEigenType pad_value = 0;

    switch (Dtype)
    {
        case TOSA_REF_TYPE_BOOL:
        case TOSA_REF_TYPE_INT8:
        case TOSA_REF_TYPE_INT16:
        case TOSA_REF_TYPE_INT32:
            pad_value = (InEigenType)attribute->pad_const_int();
            break;
        case TOSA_REF_TYPE_FP16:
        case TOSA_REF_TYPE_BF16:
        case TOSA_REF_TYPE_FP32:
        case TOSA_REF_TYPE_FP64:
        case TOSA_REF_TYPE_FP8E4M3:
        case TOSA_REF_TYPE_FP8E5M2:
            pad_value = (InEigenType)attribute->pad_const_fp();
            break;
        default:
            ASSERT_MSG(false, "TOSA_REF_TYPE %s is not supported.", EnumNameTOSAREFTYPE(Dtype));
            break;
    }

    // padding is an 1D array of [Rank * 2], with ordering:
    // [Rank0_front, Rank0_back, Rank1_front, Rank1_back, ..., Rank(N-1)_front, Rank(N-1)_back]
    TPadding padding_val = this->padding->getTensor();
    ERROR_IF(padding_val.size() != (Rank * 2), "OpPad: padding length needs to be (rank(input1) * 2)");
    for (int i = 0; i < Rank; i++)
    {
        auto pad_front = padding_val(2 * i);
        auto pad_back  = padding_val(2 * i + 1);
        ERROR_IF((pad_front < 0) || (pad_back < 0), "OpPad: padding can't be smaller than 0");
        ERROR_IF(out->getShape()[i] != pad_front + in->getShape()[i] + pad_back,
                 "OpPad: output shape not equal to input plus padding");
        paddings_array[i] = std::make_pair(pad_front, pad_back);
    }

    this->out->getTensor() = this->in->getTensor().pad(this->paddings_array, pad_value);

    return GraphNode::eval();
}

template <int Rank, TOSA_REF_TYPE Dtype>
OpDim<Rank, Dtype>::OpDim(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_)
    : GraphNode(sgt_, Op_DIM, id_)
{
    setRequiredOperands(1, 1);

    INIT_ATTRIBUTE(Axis);
}

template <int Rank, TOSA_REF_TYPE Dtype>
OpDim<Rank, Dtype>::~OpDim()
{
    if (attribute)
        delete attribute;
}

template <int Rank, TOSA_REF_TYPE Dtype>
int OpDim<Rank, Dtype>::checkTensorAttributes()
{
    // Check Tosa Level
    auto tosa_level = g_func_config.tosa_level;
    LEVEL_CHECK(Rank <= tosa_level.MAX_RANK, "Rank should be smaller than or equal to MAX_RANK");

    if (validateRequiredOperands())
        return 1;

    if (validateRequiredRank(inputs[0]))
        return 1;

    if (attribute->axis() < 0 || (size_t)attribute->axis() >= Rank)
    {
        printNodeValidationError("OpDim: axis must between [0, input_rank - 1]");
        return 1;
    }

    in  = dynamic_cast<TosaReference::TensorTemplate<TIn>*>(inputs[0]);
    out = dynamic_cast<TosaReference::TensorTemplate<TOut>*>(outputs[0]);

    ASSERT_MEM(in && out);

    return 0;
}

template <int Rank, TOSA_REF_TYPE Dtype>
int OpDim<Rank, Dtype>::eval()
{
    int32_t axis    = attribute->axis();
    int64_t out_val = in->getShape()[axis];

    this->out->getTensor().setValues({ out_val });

    return GraphNode::eval();
}

template <int InRank, int OutRank, TOSA_REF_TYPE Dtype>
OpReshape<InRank, OutRank, Dtype>::OpReshape(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_)
    : GraphNode(sgt_, Op_RESHAPE, id_)
{
    setRequiredOperands(2, 1);
}

template <int InRank, int OutRank, TOSA_REF_TYPE Dtype>
OpReshape<InRank, OutRank, Dtype>::~OpReshape()
{}

template <int InRank, int OutRank, TOSA_REF_TYPE Dtype>
int OpReshape<InRank, OutRank, Dtype>::checkTensorAttributes()
{
    // Check Tosa Level
    auto tosa_level = g_func_config.tosa_level;
    LEVEL_CHECK(InRank <= tosa_level.MAX_RANK, "InRank should be smaller than or equal to MAX_RANK");
    LEVEL_CHECK(OutRank <= tosa_level.MAX_RANK, "OutRank should be smaller than or equal to MAX_RANK");

    if (validateRequiredOperands())
        return 1;

    // output and input must be the same types
    if (inputs[0]->matchType(*outputs[0]))
    {
        printNodeValidationError("OpReshape: Input and output types must match");
        return 1;
    }

    ERROR_IF(inputs[0]->getElementCount() != outputs[0]->getElementCount(),
             "Input tensor size does not match output tensor size");

    in  = dynamic_cast<TosaReference::TensorTemplate<TIn>*>(inputs[0]);
    out = dynamic_cast<TosaReference::TensorTemplate<TOut>*>(outputs[0]);

    // note: do not assert mem on shape input, because it may be {} for reshape to scalar
    // and also, because the shape input is not actually used in eval()

    ASSERT_MEM(in && out)

    return 0;
}

template <int InRank, int OutRank, TOSA_REF_TYPE Dtype>
int OpReshape<InRank, OutRank, Dtype>::eval()
{
    for (int32_t d = 0; d < OutRank; d++)
    {
        array_shape[d]  = getOutputs()[0]->getShape()[OutRank - 1 - d];
        out_reverser[d] = OutRank - 1 - d;
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

template <int Rank, TOSA_REF_TYPE Dtype>
OpReverse<Rank, Dtype>::OpReverse(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_)
    : GraphNode(sgt_, Op_REVERSE, id_)
{
    setRequiredOperands(1, 1);
    setRequiredRank(1);

    INIT_ATTRIBUTE(Axis);
}

template <int Rank, TOSA_REF_TYPE Dtype>
OpReverse<Rank, Dtype>::~OpReverse()
{
    if (attribute)
        delete attribute;
}

template <int Rank, TOSA_REF_TYPE Dtype>
int OpReverse<Rank, Dtype>::checkTensorAttributes()
{
    // Check Tosa Level
    auto tosa_level = g_func_config.tosa_level;
    LEVEL_CHECK(Rank <= tosa_level.MAX_RANK, "Rank should be smaller than or equal to MAX_RANK");

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

template <int Rank, TOSA_REF_TYPE Dtype>
int OpReverse<Rank, Dtype>::eval()
{
    out->getTensor() = in->getTensor().reverse(reverse_array);

    return GraphNode::eval();
}

template <int Rank, TOSA_REF_TYPE Dtype>
OpSlice<Rank, Dtype>::OpSlice(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_)
    : GraphNode(sgt_, Op_SLICE, id_)
{
    setRequiredOperands(3, 1);
    setRequiredRank(1);
}

template <int Rank, TOSA_REF_TYPE Dtype>
OpSlice<Rank, Dtype>::~OpSlice()
{}

template <int Rank, TOSA_REF_TYPE Dtype>
int OpSlice<Rank, Dtype>::checkTensorAttributes()
{
    // Check Tosa Level
    auto tosa_level = g_func_config.tosa_level;
    LEVEL_CHECK(Rank <= tosa_level.MAX_RANK, "Rank should be smaller than or equal to MAX_RANK");

    if (validateRequiredOperands())
        return 1;

    if (validateRequiredRank(inputs[0]) || validateRequiredRank(outputs[0]))
    {
        return 1;
    }

    // output and input must be the same types
    if (inputs[0]->matchRankType(*outputs[0]))
    {
        printNodeValidationError("Failure to match input and output rank or type");
        return 1;
    }

    in    = dynamic_cast<TosaReference::TensorTemplate<TIn>*>(inputs[0]);
    start = dynamic_cast<TosaReference::TensorTemplate<TSlicing>*>(inputs[1]);
    size  = dynamic_cast<TosaReference::TensorTemplate<TSlicing>*>(inputs[2]);
    out   = dynamic_cast<TosaReference::TensorTemplate<TOut>*>(outputs[0]);

    ASSERT_MEM(in && out && start && size);

    return 0;
}

template <int Rank, TOSA_REF_TYPE Dtype>
int OpSlice<Rank, Dtype>::eval()
{
    TSlicing start_tensor = start->getTensor();
    TSlicing size_tensor  = size->getTensor();

    // According to https://eigen.tuxfamily.org/dox/unsupported/eigen_tensors.html
    // The type of size() is <Tensor-Type>::Index, but can always handily use it like an int.
    // However, apply explicit cast to int32_t is preferred.
    ERROR_IF(static_cast<int32_t>(start_tensor.size()) != in->getRank(),
             "OpSlice: start array length needs to be rank(input)");
    ERROR_IF(static_cast<int32_t>(size_tensor.size()) != in->getRank(),
             "OpSlice: size array length needs to be rank(input)");

    for (int32_t i = 0; i < in->getRank(); i++)
    {
        int32_t b = start_tensor(i);
        int32_t s = size_tensor(i);
        ERROR_IF(b < 0 || b >= in->getShape()[i], "OpSlice: start out of boundary");
        ERROR_IF((b + s) < 0 || (b + s) > in->getShape()[i], "OpSlice: (start+size) out of boundary");
        ERROR_IF(s <= 0, "OpSlice: output must be positive");
        ERROR_IF(s != out->getShape()[i], "OpSlice: size doesn't match output tensor dimension");
        begin_array[i] = b;
        size_array[i]  = s;
    }

    out->getTensor() = in->getTensor().slice(begin_array, size_array);

    return GraphNode::eval();
}

template <int Rank, TOSA_REF_TYPE Dtype>
OpTileBase<Rank, Dtype>::OpTileBase(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_)
    : GraphNode(sgt_, Op_TILE, id_)
{
    setRequiredOperands(2, 1);
    setRequiredRank(1);
}

template <int Rank, TOSA_REF_TYPE Dtype>
OpTileBase<Rank, Dtype>::~OpTileBase()
{}

template <int Rank, TOSA_REF_TYPE Dtype>
int OpTileBase<Rank, Dtype>::checkTensorAttributes()
{
    // Check Tosa Level
    auto tosa_level = g_func_config.tosa_level;
    LEVEL_CHECK(Rank <= tosa_level.MAX_RANK, "Rank should be smaller than or equal to MAX_RANK");

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

    in        = dynamic_cast<TosaReference::TensorTemplate<TIn>*>(inputs[0]);
    multiples = dynamic_cast<TosaReference::TensorTemplate<TInMultiples>*>(inputs[1]);
    out       = dynamic_cast<TosaReference::TensorTemplate<TOut>*>(outputs[0]);

    ASSERT_MEM(in && multiples && out);

    if (multiples->getElementCount() != Rank)
    {
        printNodeValidationError("1D list 'multiples' must have size equal to input rank");
        return 1;
    }

    return 0;
}

template <int Rank, TOSA_REF_TYPE Dtype>
int OpTile<Rank, Dtype>::eval()
{
    // primary template shouldn't be called
    FATAL_ERROR("OpTile rank=%i, dtype=%s: not implemented yet", Rank, EnumNameTOSAREFTYPE(Dtype));
}

template <TOSA_REF_TYPE Dtype>
int OpTile<1, Dtype>::eval()
{
    for (int32_t od0 = 0; od0 < this->out->getShape()[0]; od0++)
    {
        int32_t id0                 = od0 % this->in->getShape()[0];
        this->out->getTensor()(od0) = this->in->getTensor()(id0);
    }

    return GraphNode::eval();
}

template <TOSA_REF_TYPE Dtype>
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

template <TOSA_REF_TYPE Dtype>
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

template <TOSA_REF_TYPE Dtype>
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

template <TOSA_REF_TYPE Dtype>
int OpTile<5, Dtype>::eval()
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
                    int32_t id3 = od3 % this->in->getShape()[3];
                    for (int32_t od4 = 0; od4 < this->out->getShape()[4]; od4++)
                    {
                        int32_t id4 = od4 % this->in->getShape()[4];
                        this->out->getTensor()(od0, od1, od2, od3, od4) =
                            this->in->getTensor()(id0, id1, id2, id3, id4);
                    }
                }
            }
        }
    }

    return GraphNode::eval();
}

template <TOSA_REF_TYPE Dtype>
int OpTile<6, Dtype>::eval()
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
                    int32_t id3 = od3 % this->in->getShape()[3];
                    for (int32_t od4 = 0; od4 < this->out->getShape()[4]; od4++)
                    {
                        int32_t id4 = od4 % this->in->getShape()[4];
                        for (int32_t od5 = 0; od5 < this->out->getShape()[5]; od5++)
                        {
                            int32_t id5 = od5 % this->in->getShape()[5];
                            this->out->getTensor()(od0, od1, od2, od3, od4, od5) =
                                this->in->getTensor()(id0, id1, id2, id3, id4, id5);
                        }
                    }
                }
            }
        }
    }

    return GraphNode::eval();
}

template <int Rank, TOSA_REF_TYPE Dtype>
OpTranspose<Rank, Dtype>::OpTranspose(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_)
    : GraphNode(sgt_, Op_TRANSPOSE, id_)
{
    setRequiredOperands(1, 1);
    setRequiredRank(1);

    INIT_ATTRIBUTE(Transpose);
}

template <int Rank, TOSA_REF_TYPE Dtype>
OpTranspose<Rank, Dtype>::~OpTranspose()
{
    if (attribute)
        delete attribute;
}

template <int Rank, TOSA_REF_TYPE Dtype>
int OpTranspose<Rank, Dtype>::checkTensorAttributes()
{
    // Check Tosa Level
    auto tosa_level = g_func_config.tosa_level;
    LEVEL_CHECK(Rank <= tosa_level.MAX_RANK, "Rank should be smaller than or equal to MAX_RANK");

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

    in  = dynamic_cast<TosaReference::TensorTemplate<TIn>*>(inputs[0]);
    out = dynamic_cast<TosaReference::TensorTemplate<TOut>*>(outputs[0]);

    ASSERT_MEM(in && out);

    ERROR_IF(attribute->perms().size() != Rank, "OpTranspose: perms array size needs to match rank(input)");

    std::array<bool, Rank> index_used;
    index_used.fill(false);
    for (int32_t d = 0; d < Rank; d++)
    {
        int32_t index = attribute->perms()[d];
        ERROR_IF(index < 0 or index >= Rank, "OpTranspose: index out of boundary");
        ERROR_IF(index_used[index], "OpTranspose: index duplicated in perm attribute");
        index_used[index] = true;
        ERROR_IF(in->getShape()[index] != out->getShape()[d], "OpTranspose: input output shape mismatch");
        perm_array[d] = index;
    }

    return 0;
}

template <int Rank, TOSA_REF_TYPE Dtype>
int OpTranspose<Rank, Dtype>::eval()
{
    out->getTensor() = in->getTensor().shuffle(perm_array);

    return GraphNode::eval();
}

// template explicit instantiation
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpConcat, FP16)
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpConcat, BF16)
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpConcat, FP32)
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpConcat, INT8)
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpConcat, INT16)
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpConcat, INT32)
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpConcat, BOOL)
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpConcat, FP64)
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpConcat, FP8E4M3);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpConcat, FP8E5M2);

DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpPad, FP16);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpPad, BF16);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpPad, FP32);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpPad, INT8);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpPad, INT16);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpPad, INT32);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpPad, BOOL);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpPad, FP64);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpPad, FP8E4M3);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpPad, FP8E5M2);

DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpDim, FP16);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpDim, BF16);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpDim, FP32);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpDim, INT8);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpDim, INT16);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpDim, INT32);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpDim, BOOL);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpDim, FP8E4M3);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpDim, FP8E5M2);

DEF_INSTANTIATE_RESHAPE(OpReshape, FP16);
DEF_INSTANTIATE_RESHAPE(OpReshape, BF16);
DEF_INSTANTIATE_RESHAPE(OpReshape, FP32);
DEF_INSTANTIATE_RESHAPE(OpReshape, INT8);
DEF_INSTANTIATE_RESHAPE(OpReshape, INT16);
DEF_INSTANTIATE_RESHAPE(OpReshape, INT32);
DEF_INSTANTIATE_RESHAPE(OpReshape, BOOL);
DEF_INSTANTIATE_RESHAPE(OpReshape, FP64);
DEF_INSTANTIATE_RESHAPE(OpReshape, FP8E4M3);
DEF_INSTANTIATE_RESHAPE(OpReshape, FP8E5M2);

DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpReverse, FP16);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpReverse, BF16);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpReverse, FP32);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpReverse, INT8);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpReverse, INT16);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpReverse, INT32);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpReverse, BOOL);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpReverse, FP64);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpReverse, FP8E4M3);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpReverse, FP8E5M2);

DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpSlice, FP16);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpSlice, BF16);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpSlice, FP32);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpSlice, INT8);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpSlice, INT16);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpSlice, INT32);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpSlice, BOOL);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpSlice, FP64);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpSlice, FP8E4M3);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpSlice, FP8E5M2);

DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpTileBase, FP16);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpTileBase, BF16);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpTileBase, FP32);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpTileBase, INT8);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpTileBase, INT16);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpTileBase, INT32);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpTileBase, BOOL);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpTileBase, FP64);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpTileBase, FP8E4M3);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpTileBase, FP8E5M2);

DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpTile, FP16);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpTile, BF16);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpTile, FP32);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpTile, INT8);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpTile, INT16);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpTile, INT32);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpTile, BOOL);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpTile, FP64);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpTile, FP8E4M3);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpTile, FP8E5M2);

DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpTranspose, FP16);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpTranspose, BF16);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpTranspose, FP32);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpTranspose, INT8);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpTranspose, INT16);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpTranspose, INT32);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpTranspose, BOOL);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpTranspose, FP64);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpTranspose, FP8E4M3);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpTranspose, FP8E5M2);

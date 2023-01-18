
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

#include "tensor_ops.h"
#include "quant_util.h"
#include "template_types.h"
#include "half.hpp"

using namespace TosaReference;
using namespace Eigen;
using namespace tosa;

int check_pool2d_attribute(tosa::TosaPoolAttribute* attribute,
                           std::vector<int32_t> input_shape,
                           std::vector<int32_t> output_shape,
                           std::string& msg)
{
    if (attribute->pad().size() != 4)
    {
        msg = "illegal size for attribute padding";
        return 1;
    }

    if (attribute->kernel().size() != 2)
    {
        msg = "illegal size for attribute kernel";
        return 1;
    }

    if (attribute->stride().size() != 2)
    {
        msg = "illegal size for attribute stride";
        return 1;
    }

    for (int32_t i : attribute->pad())
    {
        if (i < 0)
        {
            msg = "At least one pad is smaller than zero";
            return 1;
        }
    }

    for (int32_t i : attribute->kernel())
    {
        if (i < 1)
        {
            msg = "At least one kernel dimension is smaller than one";
            return 1;
        }
    }

    for (int32_t i : attribute->stride())
    {
        if (i < 1)
        {
            msg = "At least one stride dimension is smaller than one";
            return 1;
        }
    }

    int32_t IH = input_shape[1];
    int32_t IW = input_shape[2];
    int32_t OH = output_shape[1];
    int32_t OW = output_shape[2];

    int32_t pad_top    = attribute->pad()[0];
    int32_t pad_bottom = attribute->pad()[1];
    int32_t pad_left   = attribute->pad()[2];
    int32_t pad_right  = attribute->pad()[3];

    int32_t stride_y = attribute->stride()[0];
    int32_t stride_x = attribute->stride()[1];
    int32_t kernel_y = attribute->kernel()[0];
    int32_t kernel_x = attribute->kernel()[1];

    if (pad_top >= kernel_y || pad_bottom >= kernel_y || pad_left >= kernel_x || pad_right >= kernel_x)
    {
        msg = "At least one pad is >= kernel dimension";
        return 1;
    }

    int32_t full_H = IH + pad_top + pad_bottom - kernel_y;
    int32_t full_W = IW + pad_left + pad_right - kernel_x;

    if ((full_H % stride_y != 0) ||
        (full_W % stride_x != 0))
    {
        msg = "Parameters must yield exact integer output dimensions";
        return 1;
    }

    if ((OH != (full_H / stride_y) + 1) ||
        (OW != (full_W / stride_x) + 1))
    {
        msg = "Mismatch between output shape provided and expected output shape (" +
            std::to_string((full_H / stride_y) + 1) + "," +
            std::to_string((full_W / stride_x) + 1) + ")";
        return 1;
    }

    return 0;
}

int check_conv_attribute(tosa::TosaConvAttribute* attribute,
                               uint32_t conv_dimension,
                               std::vector<int32_t> input_shape,
                               std::vector<int32_t> output_shape,
                               std::vector<int32_t> weights,
                               uint32_t offset_kernel,
                               DType InDtype,
                               DType WeightDtype,
                               std::string& msg)
{
    if (attribute->pad().size() != (2 * conv_dimension))
    {
        msg = "Illegal size for attribute pad";
        return 1;
    }

    if (attribute->stride().size() != conv_dimension)
    {
        msg = "Illegal size for attribute stride";
        return 1;
    }

    if (attribute->dilation().size() != conv_dimension)
    {
        msg = "Illegal size for attribute dilation";
        return 1;
    }

    for (int32_t i : attribute->pad())
    {
        if (i < 0)
        {
            msg = "At least one pad is smaller than zero";
            return 1;
        }
    }

    for (int32_t i : attribute->stride())
    {
        if (i < 1)
        {
            msg = "At least one stride dimension is smaller than one";
            return 1;
        }
    }

    for (int32_t i : attribute->dilation())
    {
        if (i < 1)
        {
            msg = "At least one dilation dimension is smaller than one";
            return 1;
        }
    }

    ASSERT_MSG(conv_dimension == 2 || conv_dimension == 3, "Unsupported convolution dimension")

    int32_t offset_d = conv_dimension == 3 ? 1 : 0;
    int32_t ID = conv_dimension == 3 ? input_shape[1] : 1;
    int32_t IH = input_shape[1 + offset_d];
    int32_t IW = input_shape[2 + offset_d];
    int32_t OD = conv_dimension == 3 ? output_shape[1] : 1;
    int32_t OH = output_shape[1 + offset_d];
    int32_t OW = output_shape[2 + offset_d];

    int32_t stride_d = conv_dimension == 3 ? attribute->stride()[0] : 1;
    int32_t stride_y = attribute->stride()[0 + offset_d];
    int32_t stride_x = attribute->stride()[1 + offset_d];
    int32_t kernel_d = conv_dimension == 3 ? weights[offset_kernel] : 1;
    int32_t kernel_h = weights[offset_kernel + offset_d];
    int32_t kernel_w = weights[offset_kernel + 1 + offset_d];
    int32_t dilation_d = conv_dimension == 3 ? attribute->dilation()[0] : 1;
    int32_t dilation_y = attribute->dilation()[0 + offset_d];
    int32_t dilation_x = attribute->dilation()[1 + offset_d];

    offset_d *= 2;
    int32_t pad_d0     = conv_dimension == 3 ? attribute->pad()[0] : 0;
    int32_t pad_d1     = conv_dimension == 3 ? attribute->pad()[1] : 0;
    int32_t pad_top    = attribute->pad()[0 + offset_d];
    int32_t pad_bottom = attribute->pad()[1 + offset_d];
    int32_t pad_left   = attribute->pad()[2 + offset_d];
    int32_t pad_right  = attribute->pad()[3 + offset_d];

    int32_t full_D = ID - 1 + pad_d0 + pad_d1 - (kernel_d - 1) * dilation_d;
    int32_t full_H = IH - 1 + pad_top + pad_bottom - (kernel_h - 1) * dilation_y;
    int32_t full_W = IW - 1 + pad_left + pad_right - (kernel_w - 1) * dilation_x;

    if ((full_H % stride_y != 0) ||
        (full_W % stride_x != 0) ||
        (full_D % stride_d != 0))
    {
        msg = "Parameters must yield exact integer output dimensions";
        return 1;
    }

    if ((OH != (full_H / stride_y) + 1) ||
        (OW != (full_W / stride_x) + 1) ||
        (OD != (full_D / stride_d) + 1))
    {
        std::string msg_d = "";
        if (conv_dimension == 3)
        {
            msg_d += std::to_string((full_D / stride_d) + 1) + ",";
        }
        msg = "Mismatch between output shape provided and expected output shape (" +
            msg_d +
            std::to_string((full_H / stride_y) + 1) + "," +
            std::to_string((full_W / stride_x) + 1) + ")";
        return 1;
    }

    if (InDtype != DType_INT8 && attribute->input_zp() != 0) {
        msg = "Input zero point must be zero for non-int8 data";
        return 1;
    }
    if (WeightDtype != DType_INT8 && attribute->weight_zp() != 0) {
        msg = "Weight zero point must be zero for non-int8 data";
        return 1;
    }

    return 0;
}

template <int Rank, DType Dtype>
OpArgMax<Rank, Dtype>::OpArgMax(SubgraphTraverser* sgt_,
                                TosaAttributeBase* attribute_,
                                uint64_t id_)
    : GraphNode(sgt_, Op_ARGMAX, id_)
{
    setRequiredOperands(1, 1);
    setRequiredRank(1, 4);

    INIT_ATTRIBUTE(Axis);
}

template <int Rank, DType Dtype>
OpArgMax<Rank, Dtype>::~OpArgMax()
{
    if (attribute)
        delete attribute;
}

template <int Rank, DType Dtype>
int OpArgMax<Rank, Dtype>::checkTensorAttributes()
{
    if (validateRequiredOperands())
        return 1;

    if (validateRequiredRank(inputs[0]))
    {
        return 1;
    }

    int32_t output_rank = inputs[0]->getRank() - 1;
    if (output_rank != outputs[0]->getRank())
    {
        printNodeValidationError("OpArgMax: Output rank needs to be rank(input) - 1");
        return 1;
    }

    if (outputs[0]->getDtype() != DType_INT32)
    {
        printNodeValidationError("OpArgMax: Output data type not supported for this configuration of operator");
        return 1;
    }

    input  = dynamic_cast<TosaReference::TensorTemplate<TIn>*>(inputs[0]);
    output = dynamic_cast<TosaReference::TensorTemplate<TOut>*>(outputs[0]);

    if (attribute->axis() < 0 || attribute->axis() >= input->getRank())
    {
        printNodeValidationError("OpArgMax: Axis needs to be within [0, rank(input)]");
        return 1;
    }

    bool shape_check = true;
    for (int32_t i = 0; i < input->getRank(); i++)
    {
        if (i < attribute->axis())
        {
            if (input->getShape()[i] != output->getShape()[i])
            {
                shape_check = false;
                break;
            }
        }
        else if (i > attribute->axis())
        {
            if (input->getShape()[i] != output->getShape()[i - 1])
            {
                shape_check = false;
                break;
            }
        }
        // No need to check i == axis
    }
    if (!shape_check)
    {
        printNodeValidationError("OpArgMax: Mismatch between output shape provided and expected output shape");
        return 1;
    }

    return 0;
}

template <int Rank, DType Dtype>
int OpArgMax<Rank, Dtype>::eval()
{
    Eigen::Tensor<DenseIndex, Rank - 1> index = this->input->getTensor().argmax(attribute->axis());

    this->output->getTensor() = index.unaryExpr([](DenseIndex in) -> OutEigenType { return (OutEigenType)in; });

    return GraphNode::eval();
}

template <DType Dtype, DType AccDtype>
OpAvgPool2d<Dtype, AccDtype>::OpAvgPool2d(SubgraphTraverser* sgt_,
                                TosaAttributeBase* attribute_,
                                uint64_t id_)
    : GraphNode(sgt_, Op_AVG_POOL2D, id_)
{
    setRequiredOperands(1, 1);
    setRequiredRank(4);

    INIT_ATTRIBUTE(Pool);
}

template <DType Dtype, DType AccDtype>
OpAvgPool2d<Dtype, AccDtype>::~OpAvgPool2d()
{
    if (attribute)
        delete attribute;
}

template <DType Dtype, DType AccDtype>
int OpAvgPool2d<Dtype, AccDtype>::checkTensorAttributes()
{
    if (validateRequiredOperands())
        return 1;

    if (validateRequiredRank(inputs[0]) || validateRequiredRank(outputs[0]))
    {
        return 1;
    }

    if (inputs[0]->matchType(*outputs[0]))
    {
        printNodeValidationError("OpAvgPool2d: input and output tensor type mismatch");
        return 1;
    }

    in  = dynamic_cast<TosaReference::TensorTemplate<TIn>*>(inputs[0]);
    out = dynamic_cast<TosaReference::TensorTemplate<TOut>*>(outputs[0]);

    ERROR_IF(Dtype != DType_INT8 && attribute->input_zp() != 0, "OpAvgPool2d: Input zeropoint must be zero for non int8_t data");
    ERROR_IF(Dtype != DType_INT8 && attribute->output_zp() != 0, "OpAvgPool2d: Output zeropoint must be zero for non int8_t data");

    std::string msg;
    if (check_pool2d_attribute(attribute, in->getShape(), out->getShape(), msg))
    {
        msg = "OpAvgPool2d: " + msg;
        printNodeValidationError(msg.c_str());
        return 1;
    }

    return 0;
}

// This calculates the number of padding elements used for each location along an axis
// Average pooling only divides by the number of elements used, not including padding.
// This function uses left/right, but is also used for vertical padding with top/bottom
template <DType Dtype, DType AccDtype>
ETensor1<int32_t> OpAvgPool2d<Dtype, AccDtype>::calculate_div_map_1d(int in_size, int out_size, int kernel_size, int stride, int32_t pad_left, int32_t pad_right)
{
    ETensor1<int32_t> result(out_size);

    result.setConstant(kernel_size);

    // adjust divisors on the left side for padding
    // We start at the leftmost output element, and remove pad_left - (index * stride) elements
    // until we have no more padding being used
    for(int index = 0; (index <= pad_left / stride) && (index < out_size); index++) {
        int32_t adjust = pad_left - (index * stride);
        result(index) -= adjust;
    }

    // The process repeats on the right side. Padding starts taking effect as we
    // near the rightmost input element. The first output element which touches
    // padding is defined in the initialization of index below. Then we keep moving
    // to the right, increasing padding until we get to the last output element.
    int index = std::max(0, ((pad_left + in_size - kernel_size) / stride) + 1);
    for (; index < out_size; index++) {
        int32_t adjust = ((index * stride) + kernel_size) - (pad_left + in_size);
        result(index) -= adjust;
    }
    return result;
}

// assuming input and output tensor have same scales like tflite reference
// so no need to scale input and output
template <DType Dtype, DType AccDtype>
int OpAvgPool2d<Dtype, AccDtype>::eval()
{
    int in_batch    = this->in->getShape()[0];
    int in_height   = this->in->getShape()[1];
    int in_width    = this->in->getShape()[2];
    int in_channels = this->in->getShape()[3];

    int out_batch    = this->out->getShape()[0];
    int out_height   = this->out->getShape()[1];
    int out_width    = this->out->getShape()[2];
    int out_channels = this->out->getShape()[3];

    ERROR_IF(in_batch != out_batch, "OpAvgPool2d: tensor batch mismatch %d != %d", in_batch, out_batch);
    ERROR_IF(in_channels != out_channels, "OpAvgPool2d: tensor channel mismatch %d != %d", in_channels, out_channels);

    int pad_top    = this->attribute->pad()[0];
    int pad_bottom = this->attribute->pad()[1];
    int pad_left   = this->attribute->pad()[2];
    int pad_right  = this->attribute->pad()[3];
    int kernel_h       = this->attribute->kernel()[0];
    int kernel_w       = this->attribute->kernel()[1];
    int stride_h       = this->attribute->stride()[0];
    int stride_w       = this->attribute->stride()[1];

    tosa::DType accum_dtype       = (tosa::DType)this->attribute->accum_dtype();

    DEBUG_INFO(OP,
               "perform AvgPool2d, input.shape=[%d,%d,%d,%d], output.shape=[%d,%d,%d,%d], kernel=[%d,%d], "
               "stride=[%d,%d], pad=[%d,%d,%d,%d], accum_dtype=%s",
               in_batch, in_height, in_width, in_channels, out_batch, out_height, out_width, out_channels, kernel_h,
               kernel_w, stride_h, stride_w, pad_top, pad_bottom, pad_left, pad_right, EnumNamesDType()[accum_dtype]);

    Eigen::array<Eigen::Index, 2> im2col_input_dims;
    im2col_input_dims[0] = kernel_h * kernel_w;
    im2col_input_dims[1] = out_batch * out_height * out_width * out_channels;

    Eigen::array<Eigen::Index, 4> col2im_output_dims;
    col2im_output_dims[0] = out_batch;
    col2im_output_dims[1] = out_height;
    col2im_output_dims[2] = out_width;
    col2im_output_dims[3] = out_channels;

    Eigen::array<std::pair<int32_t, int32_t>, 4> pad;
    pad[0] = std::make_pair(0, 0);
    pad[1] = std::make_pair(pad_top, pad_bottom);
    pad[2] = std::make_pair(pad_left, pad_right);
    pad[3] = std::make_pair(0, 0);

    ETensor4<InEigenType> input_val = this->in->getTensor();
    if (Dtype == DType_INT8)
    {
        input_val = input_val - (InEigenType)attribute->input_zp();
    }

    ETensor4<InEigenType> input_padded = input_val.pad(pad);

    // assuming input and output have same scales
    // so input and output scaling is not required
    // TODO: check if this assumption TOSA made

    // extract_image_patches() output [N, KH, KW, H * W, C]
    // transpose to [KH, KW, N, H * W, C]
    // reshape to [KH * KW, N * H * W * C]
    ETensor2<InEigenType> input_extract_patches =
        input_padded.extract_image_patches(kernel_h, kernel_w, stride_h, stride_w, 1, 1, Eigen::PADDING_VALID)
            .shuffle(Eigen::array<Eigen::Index, 5>{ 1, 2, 0, 3, 4 })
            .reshape(im2col_input_dims);

    // 1D result with [N * H * W * C]
    ETensor1<AccEigenType> out_1d(this->out->getElementCount());
    out_1d.setZero();

    // sum pool
    for (size_t i = 0; i < this->out->getElementCount(); i++)
    {
        for (int32_t j = 0; j < kernel_h * kernel_w; j++)
        {
            out_1d(i) += (AccEigenType)input_extract_patches(j, i);
        }
    }

    // reshape result to [N, H, W, C] and divide with div_map
    ETensor4<AccEigenType> sum = out_1d.reshape(col2im_output_dims);

    // calculate 1d height/width div_map (number of elements this pooling window covers)
    // and outer product to get 2d div_map, then reshape/broadcast to [N, H, W, C]
    ETensor1<int32_t> div_map_h = calculate_div_map_1d(in_height, out_height, kernel_h, stride_h, pad_top, pad_bottom);
    ETensor1<int32_t> div_map_w = calculate_div_map_1d(in_width, out_width, kernel_w, stride_w, pad_left, pad_right);
    Eigen::array<Eigen::IndexPair<Eigen::Index>, 1> contract_dims = { Eigen::IndexPair<Eigen::Index>(1, 0) };
    Eigen::array<Eigen::Index, 4> bcast{ out_batch, 1, 1, out_channels };

    ETensor2<int32_t> dm2_w = div_map_w.reshape(Eigen::array<Eigen::Index, 2>{ 1, out_width });
    ETensor2<int32_t> dm2_h = div_map_h.reshape(Eigen::array<Eigen::Index, 2>{ out_height, 1 });
    ETensor4<int32_t> div_map =
        dm2_h.contract(dm2_w, contract_dims)
            .reshape(Eigen::array<Eigen::Index, 4>{ 1, out_height, out_width, 1 })
            .broadcast(bcast);
    if (Dtype != DType_FP32 && Dtype != DType_FP16 && Dtype != DType_BF16)
    {
        try
        {
            this->out->getTensor() = sum.binaryExpr(div_map, [](AccEigenType value, int32_t div) -> OutEigenType {
                int32_t multiplier, shift;
                TosaReference::QuantUtil::reciprocal_scale(div, multiplier, shift);

                return (OutEigenType)TosaReference::QuantUtil::apply_scale_32(value, multiplier, shift, false);
            });
        }
        catch (std::string desc)
        {
            REQUIRE(false, "OpAvgPool2d apply_scale_32() fails: %s.", desc.c_str());
        }
        this->out->getTensor() = this->out->getTensor() + (OutEigenType)(attribute->output_zp());
        this->out->getTensor() = this->out->getTensor().cwiseMax((OutEigenType)QMin);
        this->out->getTensor() = this->out->getTensor().cwiseMin((OutEigenType)QMax);
    }
    else
    {
        // Case for float-types
        this->out->getTensor() = (sum / div_map.template cast<AccEigenType>()).template cast<OutEigenType>();
    }

    return GraphNode::eval();
}

template <DType InDtype, DType WeightDtype, DType OutDtype>
OpConv2d<InDtype, WeightDtype, OutDtype>::OpConv2d(SubgraphTraverser* sgt_,
                                         TosaAttributeBase* attribute_,
                                         uint64_t id_)
    : GraphNode(sgt_, Op_CONV2D, id_)
{
    setRequiredOperands(3, 1);
    setRequiredRank(4);

    INIT_ATTRIBUTE(Conv);
}

template <DType InDtype, DType WeightDtype, DType OutDtype>
OpConv2d<InDtype, WeightDtype, OutDtype>::~OpConv2d()
{
    if (attribute)
        delete attribute;
}

template <DType InDtype, DType WeightDtype, DType OutDtype>
int OpConv2d<InDtype, WeightDtype, OutDtype>::checkTensorAttributes()
{
    if (validateRequiredOperands())
        return 1;

    if (validateRequiredRank(inputs[0]) || validateRequiredRank(inputs[1]) || validateRequiredRank(outputs[0]))
    {
        return 1;
    }

    // 'bias' checked separatedly since it doens't make sense to make required rank ranging from 1 to 4
    if (inputs[2]->getRank() != 1)
    {
        printNodeValidationError("OpConv2d: bias tensor must be rank 1");
    }

    ERROR_IF(outputs[0]->getDtype() != OutDtype,
                "OpConv2d: Output data type not supported for this configuration of operator");

    input  = dynamic_cast<TosaReference::TensorTemplate<TIn>*>(inputs[0]);
    weight = dynamic_cast<TosaReference::TensorTemplate<TWeight>*>(inputs[1]);
    bias   = dynamic_cast<TosaReference::TensorTemplate<TBias>*>(inputs[2]);
    output = dynamic_cast<TosaReference::TensorTemplate<TOut>*>(outputs[0]);

    std::string msg;
    if (check_conv_attribute(attribute, 2 /* conv_dimension */, input->getShape(), output->getShape(),
                                   weight->getShape(), 1 /* offset_kernel */, InDtype, WeightDtype, msg))
    {
        msg = "OpConv2d: " + msg;
        printNodeValidationError(msg.c_str());
        return 1;
    }

    return 0;
}

template <DType InDtype, DType WeightDtype, DType OutDtype>
int OpConv2d<InDtype, WeightDtype, OutDtype>::eval()
{
    int in_batch    = this->input->getShape()[0];
    int in_height   = this->input->getShape()[1];
    int in_width    = this->input->getShape()[2];
    int in_channels = this->input->getShape()[3];

    int f_out_channels = this->weight->getShape()[0];
    int f_height       = this->weight->getShape()[1];
    int f_width        = this->weight->getShape()[2];
    int f_in_channels  = this->weight->getShape()[3];

    int b_out_channels = this->bias->getShape()[0];

    int out_batch    = this->output->getShape()[0];
    int out_height   = this->output->getShape()[1];
    int out_width    = this->output->getShape()[2];
    int out_channels = this->output->getShape()[3];

    ERROR_IF(in_batch != out_batch, "OpConv2d: tensor batch mismatch %d != %d", in_batch, out_batch);
    ERROR_IF(f_in_channels != in_channels, "OpConv2d: tensor input channel mismatch %d != %d", f_in_channels,
             in_channels);
    ERROR_IF(f_out_channels != out_channels, "OpConv2d: tensor output channel mismatch %d != %d", f_out_channels,
             out_channels);
    ERROR_IF(b_out_channels != out_channels, "OpConv2d: bias channel mismatch %d != %d", b_out_channels, out_channels);

    int pad_top    = this->attribute->pad()[0];
    int pad_bottom = this->attribute->pad()[1];
    int pad_left   = this->attribute->pad()[2];
    int pad_right  = this->attribute->pad()[3];

    int stride_h       = this->attribute->stride()[0];
    int stride_w       = this->attribute->stride()[1];
    int dilation_h     = this->attribute->dilation()[0];
    int dilation_w     = this->attribute->dilation()[1];

    DEBUG_INFO(OP,
               "perform OpConv2d, input.shape=[%d,%d,%d,%d], weight.shape=[%d,%d,%d,%d], output.shape=[%d,%d,%d,%d], "
               "stride=[%d,%d], dilation=[%d,%d], pad=[%d,%d,%d,%d]",
               in_batch, in_height, in_width, in_channels, f_height, f_width, f_in_channels, f_out_channels, out_batch,
               out_height, out_width, out_channels, stride_h, stride_w, dilation_h, dilation_w, pad_top,
               pad_bottom, pad_left, pad_right);

    // GEMM-conv2d, left matrix is input, right matrix is weight
    Eigen::array<Eigen::Index, 2> im2col_input_dims;
    im2col_input_dims[0] = out_batch * out_height * out_width;
    im2col_input_dims[1] = f_height * f_width * f_in_channels;

    Eigen::array<Eigen::Index, 2> im2col_weight_dims;
    im2col_weight_dims[0] = f_height * f_width * f_in_channels;
    im2col_weight_dims[1] = f_out_channels;

    Eigen::array<Eigen::Index, 2> bias_reshaped_dims;
    bias_reshaped_dims[0] = 1;
    bias_reshaped_dims[1] = b_out_channels;

    Eigen::array<Eigen::Index, 4> weight_zp_bcast_dims;
    weight_zp_bcast_dims[0] = f_height;
    weight_zp_bcast_dims[1] = f_width;
    weight_zp_bcast_dims[2] = f_in_channels;

    Eigen::array<Eigen::Index, 2> bias_bcast_dims;
    bias_bcast_dims[0] = out_batch * out_height * out_width;
    bias_bcast_dims[1] = 1;

    Eigen::array<Eigen::Index, 4> col2im_output_dims;
    col2im_output_dims[0] = out_batch;
    col2im_output_dims[1] = out_height;
    col2im_output_dims[2] = out_width;
    col2im_output_dims[3] = out_channels;

    Eigen::array<Eigen::IndexPair<Eigen::Index>, 1> contract_dims = { Eigen::IndexPair<Eigen::Index>(1, 0) };

    Eigen::array<std::pair<int32_t, int32_t>, 4> pad;
    pad[0] = std::make_pair(0, 0);
    pad[1] = std::make_pair(pad_top, pad_bottom);
    pad[2] = std::make_pair(pad_left, pad_right);
    pad[3] = std::make_pair(0, 0);

    TIn input_val      = this->input->getTensor();
    TWeight weight_val = this->weight->getTensor();
    if (InDtype == DType_INT8 || WeightDtype == DType_INT8)
    {
        input_val  = input_val - (InEigenType)attribute->input_zp();
        weight_val = weight_val - (WeightEigenType)attribute->weight_zp();
    }

    ETensor4<InEigenType> input_padded = input_val.pad(pad);

    // extract_image_patches() output [N, KH, KW, H * W, C]
    // need to transpose to [N, H * W, KH, KW, C]
    ETensor5<InEigenType> input_extract_patches =
        input_padded
            .extract_image_patches(f_height, f_width, stride_h, stride_w, dilation_h, dilation_w, Eigen::PADDING_VALID)
            .shuffle(Eigen::array<Eigen::Index, 5>{ 0, 3, 1, 2, 4 });

    // reshape input to [N * H * W, KH * KW * C]
    ETensor2<InEigenType> im2col_input = input_extract_patches.reshape(im2col_input_dims);

    // transpose and reshape weight from [OC, H, W, IC] to [H * W * IC, OC]
    ETensor2<WeightEigenType> im2col_weight =
    weight_val.shuffle(Eigen::array<Eigen::Index, 4>({ 1, 2, 3, 0 })).reshape(im2col_weight_dims);

    // don't need to apply bias_multiplier ( * bias_scale and >> bias_shift) since tflite already scale it
    // and reshaped from [C] to [1, C], and broadcast to [N * H * W, C]
    ETensor2<OutEigenType> bias_2d = (this->bias->getTensor().reshape(bias_reshaped_dims).broadcast(bias_bcast_dims)).template cast<OutEigenType>();

    // output matrix is [N * H * W, C]
    ETensor2<OutEigenType> contracted_result =
        (im2col_input.template cast<AccEigenType>().contract(im2col_weight.template cast<AccEigenType>(), contract_dims)).template cast<OutEigenType>();

    // adding bias
    ETensor2<OutEigenType> biased_output = contracted_result + bias_2d;

    // reshape back to [N, H, W, C]
    this->output->getTensor() = biased_output.reshape(col2im_output_dims);

    if (OutDtype == DType_INT48)
    {
        this->output->getTensor() = this->output->getTensor().cwiseMax((OutEigenType)AccQMin);
        this->output->getTensor() = this->output->getTensor().cwiseMin((OutEigenType)AccQMax);
    }

    return GraphNode::eval();
}

template <DType InDtype, DType WeightDtype, DType OutDtype>
OpConv3d<InDtype, WeightDtype, OutDtype>::OpConv3d(SubgraphTraverser* sgt_,
                                         TosaAttributeBase* attribute_,
                                         uint64_t id_)
    : GraphNode(sgt_, Op_CONV3D, id_)
{
    setRequiredOperands(3, 1);
    setRequiredRank(5);

    INIT_ATTRIBUTE(Conv);
}

template <DType InDtype, DType WeightDtype, DType OutDtype>
OpConv3d<InDtype, WeightDtype, OutDtype>::~OpConv3d()
{
    if (attribute)
        delete attribute;
}

template <DType InDtype, DType WeightDtype, DType OutDtype>
int OpConv3d<InDtype, WeightDtype, OutDtype>::checkTensorAttributes()
{
    if (validateRequiredOperands())
        return 1;

    if (validateRequiredRank(inputs[0]) || validateRequiredRank(inputs[1]) || validateRequiredRank(outputs[0]))
    {
        return 1;
    }

    // 'bias' checked separatedly since it doens't make sense to make required rank ranging from 1 to 4
    if (inputs[2]->getRank() != 1)
    {
        printNodeValidationError("OpConv3d: bias tensor must be rank 1");
    }

    ERROR_IF(outputs[0]->getDtype() != OutDtype,
                "OpConv3d: Output data type not supported for this configuration of operator");

    input  = dynamic_cast<TosaReference::TensorTemplate<TIn>*>(inputs[0]);
    weight = dynamic_cast<TosaReference::TensorTemplate<TWeight>*>(inputs[1]);
    bias   = dynamic_cast<TosaReference::TensorTemplate<TBias>*>(inputs[2]);
    output = dynamic_cast<TosaReference::TensorTemplate<TOut>*>(outputs[0]);

    std::string msg;
    if (check_conv_attribute(attribute, 3 /* conv_dimension */, input->getShape(), output->getShape(),
                                   weight->getShape(), 1 /* offset_kernel */, InDtype, WeightDtype, msg))
    {
        msg = "OpConv3d: " + msg;
        printNodeValidationError(msg.c_str());
        return 1;
    }

    return 0;
}

template <DType InDtype, DType WeightDtype, DType OutDtype>
int OpConv3d<InDtype, WeightDtype, OutDtype>::eval()
{
    int in_batch    = this->input->getShape()[0];
    int in_depth    = this->input->getShape()[1];
    int in_height   = this->input->getShape()[2];
    int in_width    = this->input->getShape()[3];
    int in_channels = this->input->getShape()[4];

    int f_out_channels = this->weight->getShape()[0];
    int f_depth        = this->weight->getShape()[1];
    int f_height       = this->weight->getShape()[2];
    int f_width        = this->weight->getShape()[3];
    int f_in_channels  = this->weight->getShape()[4];

    int b_out_channels = this->bias->getShape()[0];

    int out_batch    = this->output->getShape()[0];
    int out_depth    = this->output->getShape()[1];
    int out_height   = this->output->getShape()[2];
    int out_width    = this->output->getShape()[3];
    int out_channels = this->output->getShape()[4];

    ERROR_IF(in_batch != out_batch, "OpConv3d: tensor batch mismatch %d != %d", in_batch, out_batch);
    ERROR_IF(f_in_channels != in_channels, "OpConv3d: tensor input channel mismatch %d != %d", f_in_channels,
             in_channels);
    ERROR_IF(f_out_channels != out_channels, "OpConv3d: tensor output channel mismatch %d != %d", f_out_channels,
             out_channels);
    ERROR_IF(b_out_channels != out_channels, "OpConv3d: bias channel mismatch %d != %d", b_out_channels, out_channels);

    int pad_d0     = this->attribute->pad()[0];
    int pad_d1     = this->attribute->pad()[1];
    int pad_top    = this->attribute->pad()[2];
    int pad_bottom = this->attribute->pad()[3];
    int pad_left   = this->attribute->pad()[4];
    int pad_right  = this->attribute->pad()[5];

    int stride_d       = this->attribute->stride()[0];
    int stride_h       = this->attribute->stride()[1];
    int stride_w       = this->attribute->stride()[2];

    int dilation_d     = this->attribute->dilation()[0];
    int dilation_h     = this->attribute->dilation()[1];
    int dilation_w     = this->attribute->dilation()[2];

    DEBUG_INFO(
        OP,
        "perform OpConv3d, input.shape=[%d,%d,%d,%d,%d], weight.shape=[%d,%d,%d,%d,%d], output.shape=[%d,%d,%d,%d,%d], "
        "stride=[%d,%d,%d], dilation=[%d,%d,%d], pad=[%d,%d,%d,%d,%d,%d]",
        in_batch, in_depth, in_height, in_width, in_channels, f_out_channels, f_depth, f_height, f_width, f_in_channels,
        out_batch, out_depth, out_height, out_width, out_channels, stride_d, stride_h, stride_w, dilation_d, dilation_h,
        dilation_w, pad_d0, pad_d1, pad_top, pad_bottom, pad_left, pad_right);

    Eigen::array<std::pair<int32_t, int32_t>, 5> pad;
    pad[0] = std::make_pair(0, 0);
    pad[1] = std::make_pair(pad_d0, pad_d1);
    pad[2] = std::make_pair(pad_top, pad_bottom);
    pad[3] = std::make_pair(pad_left, pad_right);
    pad[4] = std::make_pair(0, 0);

    TIn input_val      = this->input->getTensor();
    TWeight weight_val = this->weight->getTensor();
    if (InDtype == DType_INT8 || WeightDtype == DType_INT8)
    {
        input_val  = input_val - (InEigenType)attribute->input_zp();
        weight_val = weight_val - (WeightEigenType)attribute->weight_zp();
    }

    ETensor5<InEigenType> input_padded = input_val.pad(pad);

    // 1. initialize with bias
    Eigen::array<Eigen::Index, 5> reshape_dim;
    reshape_dim.fill(1);
    reshape_dim[4] = b_out_channels;

    Eigen::array<Eigen::Index, 5> bcast;
    bcast[0]                  = out_batch;
    bcast[1]                  = out_depth;
    bcast[2]                  = out_height;
    bcast[3]                  = out_width;
    bcast[4]                  = 1;
    this->output->getTensor() = this->bias->getTensor().reshape(reshape_dim).broadcast(bcast);

    // 2. direct convolution
    AccEigenType acc(0.0);
    int d_idx, h_idx, w_idx;

    for (int ob = 0; ob < out_batch; ob++)
    {
        for (int od = 0; od < out_depth; od++)
        {
            for (int oh = 0; oh < out_height; oh++)
            {
                for (int ow = 0; ow < out_width; ow++)
                {
                    for (int oc = 0; oc < out_channels; oc++)
                    {
                        // Initialize accumulator with bias value
                        acc = (AccEigenType)this->output->getTensor()(ob, od, oh, ow, oc);
                        for (int fd = 0; fd < f_depth; fd++)
                        {
                            d_idx = od * stride_d + fd * dilation_d;
                            for (int fh = 0; fh < f_height; fh++)
                            {
                                h_idx = oh * stride_h + fh * dilation_h;
                                for (int fw = 0; fw < f_width; fw++)
                                {
                                    w_idx = ow * stride_w + fw * dilation_w;
                                    for (int ic = 0; ic < in_channels; ic++)
                                    {
                                        acc += ((AccEigenType)input_padded(ob, d_idx, h_idx, w_idx, ic) *
                                                (AccEigenType)weight_val(oc, fd, fh, fw, ic));
                                    }
                                }
                            }
                        }
                        this->output->getTensor()(ob, od, oh, ow, oc) = (OutEigenType)acc;
                    }
                }
            }
        }
    }

    if (OutDtype == DType_INT48)
    {
        this->output->getTensor() = this->output->getTensor().cwiseMax((OutEigenType)AccQMin);
        this->output->getTensor() = this->output->getTensor().cwiseMin((OutEigenType)AccQMax);
    }

    return GraphNode::eval();
}

template <DType InDtype, DType WeightDtype, DType OutDtype>
OpDepthwiseConv2d<InDtype, WeightDtype, OutDtype>::OpDepthwiseConv2d(SubgraphTraverser* sgt_,
                                                           TosaAttributeBase* attribute_,
                                                           uint64_t id_)
    : GraphNode(sgt_, Op_DEPTHWISE_CONV2D, id_)
{
    setRequiredOperands(3, 1);
    setRequiredRank(4);

    INIT_ATTRIBUTE(Conv);
}

template <DType InDtype, DType WeightDtype, DType OutDtype>
OpDepthwiseConv2d<InDtype, WeightDtype, OutDtype>::~OpDepthwiseConv2d()
{
    if (attribute)
        delete attribute;
}

template <DType InDtype, DType WeightDtype, DType OutDtype>
int OpDepthwiseConv2d<InDtype, WeightDtype, OutDtype>::checkTensorAttributes()
{
    if (validateRequiredOperands())
        return 1;

    if (validateRequiredRank(inputs[0]) || validateRequiredRank(inputs[1]) || validateRequiredRank(outputs[0]))
    {
        return 1;
    }

    // 'bias' checked separatedly since it doens't make sense to make required rank ranging from 1 to 4
    if (inputs[2]->getRank() != 1)
    {
        printNodeValidationError("OpDepthwiseConv2d: bias tensor must be rank 1");
    }

    ERROR_IF(outputs[0]->getDtype() != OutDtype,
                "OpDepthwiseConv2d: Output data type not supported for this configuration of operator");

    input  = dynamic_cast<TosaReference::TensorTemplate<TIn>*>(inputs[0]);
    weight = dynamic_cast<TosaReference::TensorTemplate<TWeight>*>(inputs[1]);
    bias   = dynamic_cast<TosaReference::TensorTemplate<TBias>*>(inputs[2]);
    output = dynamic_cast<TosaReference::TensorTemplate<TOut>*>(outputs[0]);

    std::string msg;
    if (check_conv_attribute(attribute, 2 /* conv_dimension */, input->getShape(), output->getShape(),
                                   weight->getShape(), 0 /* offset_kernel */, InDtype, WeightDtype, msg))
    {
        msg = "OpDepthwiseConv2d: " + msg;
        printNodeValidationError(msg.c_str());
        return 1;
    }

    return 0;
}

template <DType InDtype, DType WeightDtype, DType OutDtype>
int OpDepthwiseConv2d<InDtype, WeightDtype, OutDtype>::eval()
{
    int in_batch    = this->input->getShape()[0];
    int in_height   = this->input->getShape()[1];
    int in_width    = this->input->getShape()[2];
    int in_channels = this->input->getShape()[3];

    int f_height      = this->weight->getShape()[0];
    int f_width       = this->weight->getShape()[1];
    int f_in_channels = this->weight->getShape()[2];
    int f_multiplier  = this->weight->getShape()[3];

    int b_out_channels = this->bias->getShape()[0];

    int out_batch    = this->output->getShape()[0];
    int out_height   = this->output->getShape()[1];
    int out_width    = this->output->getShape()[2];
    int out_channels = this->output->getShape()[3];

    ERROR_IF(in_batch != out_batch, "OpDepthwiseConv2d: tensor batch mismatch %d != %d", in_batch, out_batch);
    ERROR_IF(f_in_channels != in_channels, "OpDepthwiseConv2d: tensor input channel mismatch %d != %d", f_in_channels,
             in_channels);
    ERROR_IF(in_channels * f_multiplier != out_channels, "OpDepthwiseConv2d: tensor output channel mismatch %d != %d",
             in_channels * f_multiplier, out_channels);
    ERROR_IF(b_out_channels != out_channels, "OpDepthwiseConv2d: bias channels mismatch %d != %d", b_out_channels,
             out_channels);

    int pad_top    = this->attribute->pad()[0];
    int pad_bottom = this->attribute->pad()[1];
    int pad_left   = this->attribute->pad()[2];
    int pad_right  = this->attribute->pad()[3];

    int stride_h       = this->attribute->stride()[0];
    int stride_w       = this->attribute->stride()[1];
    int dilation_h     = this->attribute->dilation()[0];
    int dilation_w     = this->attribute->dilation()[1];

    DEBUG_INFO(OP,
               "perform OpDepthwiseConv2d, input.shape=[%d,%d,%d,%d], weight.shape=[%d,%d,%d,%d], "
               "output.shape=[%d,%d,%d,%d], stride=[%d,%d], dilation=[%d,%d], pad=[%d,%d,%d,%d]",
               in_batch, in_height, in_width, in_channels, f_height, f_width, f_in_channels, f_multiplier, out_batch,
               out_height, out_width, out_channels, stride_h, stride_w, dilation_h, dilation_w, pad_top,
               pad_bottom, pad_left, pad_right);

    Eigen::array<std::pair<int32_t, int32_t>, 4> pad;
    pad[0] = std::make_pair(0, 0);
    pad[1] = std::make_pair(pad_top, pad_bottom);
    pad[2] = std::make_pair(pad_left, pad_right);
    pad[3] = std::make_pair(0, 0);

    TIn input_val      = this->input->getTensor();
    TWeight weight_val = this->weight->getTensor();
    if (InDtype == DType_INT8 || WeightDtype == DType_INT8)
    {
        input_val  = input_val - (InEigenType)attribute->input_zp();
        weight_val = weight_val - (WeightEigenType)attribute->weight_zp();
    }

    ETensor4<InEigenType> input_padded = input_val.pad(pad);

    // GEMM doesn't fit well with DepthwiseConv2d
    // 1. use extract_image_patches() to handle stride/dilation/pad
    // 2. perform direct convolution

    // 1. extract_image_patches() output [N, KH, KW, OH * OW, IC]
    ETensor5<InEigenType> input_extract_patches = input_padded.extract_image_patches(
        f_height, f_width, stride_h, stride_w, dilation_h, dilation_w, Eigen::PADDING_VALID);

    Eigen::array<Eigen::Index, 4> reshape_dim;
    reshape_dim.fill(1);
    reshape_dim[3] = b_out_channels;

    Eigen::array<Eigen::Index, 4> bcast;
    bcast[0] = out_batch;
    bcast[1] = out_height;
    bcast[2] = out_width;
    bcast[3] = 1;

    // initialize with bias
    this->output->getTensor() = this->bias->getTensor().reshape(reshape_dim).broadcast(bcast);

    // 2. direct depthwise convolution
    for (int ob = 0; ob < out_batch; ob++)
    {
        for (int oh = 0; oh < out_height; oh++)
        {
            for (int ow = 0; ow < out_width; ow++)
            {
                for (int ic = 0; ic < in_channels; ic++)
                {
                    for (int cm = 0; cm < f_multiplier; cm++)
                    {
                        for (int fh = 0; fh < f_height; fh++)
                        {
                            for (int fw = 0; fw < f_width; fw++)
                            {
                                // Perform multiplication in AccEigenType then cast to OutEigenType
                                this->output->getTensor()(ob, oh, ow, ic * f_multiplier + cm) +=
                                (OutEigenType)((AccEigenType)input_extract_patches(ob, fh, fw, ow * out_height + oh, ic) *
                                (AccEigenType)weight_val(fh, fw, ic, cm));
                            }
                        }
                    }
                }
            }
        }
    }

    if (OutDtype == DType_INT48)
    {
        this->output->getTensor() = this->output->getTensor().cwiseMax((OutEigenType)AccQMin);
        this->output->getTensor() = this->output->getTensor().cwiseMin((OutEigenType)AccQMax);
    }

    return GraphNode::eval();
}

template <DType InDtype, DType WeightDtype, DType OutDtype>
OpFullyConnected<InDtype, WeightDtype, OutDtype>::OpFullyConnected(SubgraphTraverser* sgt_,
                                                         TosaAttributeBase* attribute_,
                                                         uint64_t id_)
    : GraphNode(sgt_, Op_FULLY_CONNECTED, id_)
{
    setRequiredOperands(3, 1);
    setRequiredRank(2);

    INIT_ATTRIBUTE(FullyConnected);
}

template <DType InDtype, DType WeightDtype, DType OutDtype>
OpFullyConnected<InDtype, WeightDtype, OutDtype>::~OpFullyConnected()
{
    if (attribute)
        delete attribute;
}

template <DType InDtype, DType WeightDtype, DType OutDtype>
int OpFullyConnected<InDtype, WeightDtype, OutDtype>::checkTensorAttributes()
{
    if (validateRequiredOperands())
        return 1;

    if (validateRequiredRank(inputs[0]) || validateRequiredRank(inputs[1]) || validateRequiredRank(outputs[0]))
    {
        return 1;
    }

    input  = dynamic_cast<TosaReference::TensorTemplate<TIn>*>(inputs[0]);
    weight = dynamic_cast<TosaReference::TensorTemplate<TWeight>*>(inputs[1]);
    bias   = dynamic_cast<TosaReference::TensorTemplate<TBias>*>(inputs[2]);

    if (input->getShape()[1] != weight->getShape()[1])
    {
        printNodeValidationError("OpFullyConnected operator input.shape[1] should match weight.shape[1]");
        return 1;
    }

    if (weight->getShape()[0] != bias->getShape()[0])
    {
        printNodeValidationError("OpFullyConnected operator bias.shape[0] should match weight.shape[0]");
        return 1;
    }

    ERROR_IF(outputs[0]->getDtype() != OutDtype,
                "OpFullyConnected: Output data type not supported for this configuration of operator");

    output = dynamic_cast<TosaReference::TensorTemplate<TOut>*>(outputs[0]);

    ERROR_IF(InDtype != DType_INT8 && attribute->input_zp() != 0, "OpFullyConnected: Input zeropoint must be zero for non int8_t data");
    ERROR_IF(WeightDtype != DType_INT8 && attribute->weight_zp() != 0, "OpFullyConnected: Weight zeropoint must be zero for non int8_t data");

    return 0;
}

template <DType InDtype, DType WeightDtype, DType OutDtype>
int OpFullyConnected<InDtype, WeightDtype, OutDtype>::eval()
{
    typedef Eigen::Tensor<int, 1>::DimensionPair DimPair;
    Eigen::array<DimPair, 1> dims{ { DimPair(1, 0) } };

    Eigen::array<Eigen::Index, 2> weight_shuffle{ 1, 0 };

    Eigen::array<Eigen::Index, 2> bias_reshape;
    bias_reshape[0] = 1;
    bias_reshape[1] = this->bias->getShape()[0];

    Eigen::array<Eigen::Index, 2> bias_bcast;
    bias_bcast[0] = this->input->getShape()[0];
    bias_bcast[1] = 1;

    TIn input_val      = this->input->getTensor();
    TWeight weight_val = this->weight->getTensor().shuffle(weight_shuffle);
    if (InDtype == DType_INT8 || WeightDtype == DType_INT8)
    {
        input_val  = input_val - (InEigenType)attribute->input_zp();
        weight_val = weight_val - (WeightEigenType)attribute->weight_zp();
    }

    this->output->getTensor() =
        input_val.template cast<AccEigenType>().contract(weight_val.template cast<AccEigenType>(), dims).template cast<OutEigenType>() +
            this->bias->getTensor().reshape(bias_reshape).broadcast(bias_bcast);

    if (OutDtype == DType_INT48)
    {
        this->output->getTensor() = this->output->getTensor().cwiseMax((OutEigenType)AccQMin);
        this->output->getTensor() = this->output->getTensor().cwiseMin((OutEigenType)AccQMax);
    }
    return GraphNode::eval();
}

template <DType Dtype, DType OutDtype>
OpMatMul<Dtype, OutDtype>::OpMatMul(SubgraphTraverser* sgt_,
                          TosaAttributeBase* attribute_,
                          uint64_t id_)
    : GraphNode(sgt_, Op_MATMUL, id_)
{
    setRequiredOperands(2, 1);
    setRequiredRank(3);

    INIT_ATTRIBUTE(MatMul);
}

template <DType Dtype, DType OutDtype>
OpMatMul<Dtype, OutDtype>::~OpMatMul()
{
    if (attribute)
        delete attribute;
}

template <DType Dtype, DType OutDtype>
int OpMatMul<Dtype, OutDtype>::checkTensorAttributes()
{
    if (validateRequiredOperands())
        return 1;

    if (validateRequiredRank(inputs[0]) || validateRequiredRank(inputs[1]) || validateRequiredRank(outputs[0]))
    {
        return 1;
    }

    ERROR_IF(outputs[0]->getDtype() != OutDtype,
                "OpMatMul: Output data type not supported for this configuration of operator");

    a      = dynamic_cast<TosaReference::TensorTemplate<TIn>*>(inputs[0]);
    b      = dynamic_cast<TosaReference::TensorTemplate<TIn>*>(inputs[1]);
    output = dynamic_cast<TosaReference::TensorTemplate<TOut>*>(outputs[0]);

    ASSERT_MEM(a && b && output);

    // a: [N, H, C]
    // b: [N, C, W]
    // c: [N, H, W]

    // Check N
    if (a->getShape()[0] != b->getShape()[0] || a->getShape()[0] != output->getShape()[0])
    {
        printNodeValidationError("OpMatMul operator a.shape[0], b.shape[0] and output.shape[0] should match");
        return 1;
    }
    N = a->getShape()[0];

    // Check C
    if (a->getShape()[2] != b->getShape()[1])
    {
        printNodeValidationError("OpMatMul operator a.shape[2] should match b.shape[1]");
        return 1;
    }
    C = a->getShape()[2];

    // Check H
    if (a->getShape()[1] != output->getShape()[1])
    {
        printNodeValidationError("OpMatMul operator a.shape[1] should match output.shape[1]");
        return 1;
    }
    H = a->getShape()[1];

    // Check W
    if (b->getShape()[2] != output->getShape()[2])
    {
        printNodeValidationError("OpMatMul operator output.shape[2] should match output.shape[2]");
        return 1;
    }
    W = b->getShape()[2];

    ERROR_IF(Dtype != DType_INT8 && attribute->a_zp() != 0, "OpMatMul: A zeropoint must be zero for non int8_t data");
    ERROR_IF(Dtype != DType_INT8 && attribute->b_zp() != 0, "OpMatMul: B zeropoint must be zero for non int8_t data");

    return 0;
}

template <DType Dtype, DType OutDtype>
int OpMatMul<Dtype, OutDtype>::eval()
{
    typedef Eigen::Tensor<int, 1>::DimensionPair DimPair;
    Eigen::array<DimPair, 1> dims{ { DimPair(1, 0) } };

    TIn a_val = this->a->getTensor();
    TIn b_val = this->b->getTensor();
    if (Dtype == DType_INT8)
    {
        a_val = a_val - (InEigenType)attribute->a_zp();
        b_val = b_val - (InEigenType)attribute->b_zp();
    }

    Eigen::array<Eigen::Index, 2> a_rank2_shape({ H, C });
    Eigen::array<Eigen::Index, 2> b_rank2_shape({ C, W });
    Eigen::array<Eigen::Index, 3> output_rank3_shape({ 1, H, W });

    Eigen::array<Eigen::Index, 3> a_size_array({ 1, H, C });
    Eigen::array<Eigen::Index, 3> b_size_array({ 1, C, W });

    Eigen::array<Eigen::Index, 3> a_begin_array({ 0, 0, 0 });
    Eigen::array<Eigen::Index, 3> b_begin_array({ 0, 0, 0 });

    // Iterate N dimension.
    for (int i = 0; i < N; i++)
    {
        a_begin_array[0] = i;
        b_begin_array[0] = i;

        TInRank2 a_rank2_val = a_val.slice(a_begin_array, a_size_array).reshape(a_rank2_shape);
        TInRank2 b_rank2_val = b_val.slice(b_begin_array, b_size_array).reshape(b_rank2_shape);
        TAccRank2 output_rank2_val =
            a_rank2_val.template cast<AccEigenType>().contract(b_rank2_val.template cast<AccEigenType>(), dims);
        TOut output_rank3_val = output_rank2_val.reshape(output_rank3_shape).template cast<OutEigenType>();
        if (i == 0)
        {
            this->output->getTensor() = output_rank3_val;
        }
        else
        {
            TOut temp                 = this->output->getTensor().concatenate(output_rank3_val, 0);
            this->output->getTensor() = temp;
        }
    }

    if (OutDtype == DType_INT48)
    {
        this->output->getTensor() = this->output->getTensor().cwiseMax((OutEigenType)AccQMin);
        this->output->getTensor() = this->output->getTensor().cwiseMin((OutEigenType)AccQMax);
    }

    return GraphNode::eval();
}

template <DType Dtype>
OpMaxPool2d<Dtype>::OpMaxPool2d(SubgraphTraverser* sgt_,
                                TosaAttributeBase* attribute_,
                                uint64_t id_)
    : GraphNode(sgt_, Op_MAX_POOL2D, id_)
{
    setRequiredOperands(1, 1);
    setRequiredRank(4);

    INIT_ATTRIBUTE(Pool);
}

template <DType Dtype>
OpMaxPool2d<Dtype>::~OpMaxPool2d()
{
    if (attribute)
        delete attribute;
}

template <DType Dtype>
int OpMaxPool2d<Dtype>::checkTensorAttributes()
{
    if (validateRequiredOperands())
        return 1;

    if (validateRequiredRank(inputs[0]) || validateRequiredRank(outputs[0]))
    {
        return 1;
    }

    if (inputs[0]->matchType(*outputs[0]))
    {
        printNodeValidationError("OpMaxPool2d: input and output tensor type mismatch");
        return 1;
    }

    in  = dynamic_cast<TosaReference::TensorTemplate<TIn>*>(inputs[0]);
    out = dynamic_cast<TosaReference::TensorTemplate<TOut>*>(outputs[0]);

    std::string msg;
    if (check_pool2d_attribute(attribute, in->getShape(), out->getShape(), msg))
    {
        msg = "OpMaxPool2d: " + msg;
        printNodeValidationError(msg.c_str());
        return 1;
    }

    return 0;
}

template <DType Dtype>
int OpMaxPool2d<Dtype>::eval()
{
    int in_batch    = this->in->getShape()[0];
    int in_height   = this->in->getShape()[1];
    int in_width    = this->in->getShape()[2];
    int in_channels = this->in->getShape()[3];

    int out_batch    = this->out->getShape()[0];
    int out_height   = this->out->getShape()[1];
    int out_width    = this->out->getShape()[2];
    int out_channels = this->out->getShape()[3];

    ERROR_IF(in_batch != out_batch, "OpMaxPool2d: tensor batch mismatch %d != %d", in_batch, out_batch);
    ERROR_IF(in_channels != out_channels, "OpMaxPool2d: tensor channel mismatch %d != %d", in_channels, out_channels);

    int pad_top    = this->attribute->pad()[0];
    int pad_bottom = this->attribute->pad()[1];
    int pad_left   = this->attribute->pad()[2];
    int pad_right  = this->attribute->pad()[3];

    int kernel_h       = this->attribute->kernel()[0];
    int kernel_w       = this->attribute->kernel()[1];
    int stride_h       = this->attribute->stride()[0];
    int stride_w       = this->attribute->stride()[1];

    DEBUG_INFO(OP,
               "perform MaxPool2d, input.shape=[%d,%d,%d,%d], output.shape=[%d,%d,%d,%d], kernel=[%d,%d], "
               "stride=[%d,%d], pad=[%d,%d,%d,%d]",
               in_batch, in_height, in_width, in_channels, out_batch, out_height, out_width, out_channels, kernel_h,
               kernel_w, stride_h, stride_w, pad_top, pad_bottom, pad_left, pad_right);

    Eigen::array<Eigen::Index, 2> im2col_input_dims;
    im2col_input_dims[0] = kernel_h * kernel_w;
    im2col_input_dims[1] = out_batch * out_height * out_width * out_channels;

    Eigen::array<Eigen::Index, 4> col2im_output_dims;
    col2im_output_dims[0] = out_batch;
    col2im_output_dims[1] = out_height;
    col2im_output_dims[2] = out_width;
    col2im_output_dims[3] = out_channels;

    Eigen::array<std::pair<int32_t, int32_t>, 4> pad;
    pad[0] = std::make_pair(0, 0);
    pad[1] = std::make_pair(pad_top, pad_bottom);
    pad[2] = std::make_pair(pad_left, pad_right);
    pad[3] = std::make_pair(0, 0);

    ETensor4<InEigenType> input_padded = this->in->getTensor().pad(pad, std::numeric_limits<InEigenType>::lowest());

    // extract_image_patches() output [N, KH, KW, H * W, C]
    // transpose to [KH, KW, N, H * W, C]
    // reshape to [KH * KW, N * H * W * C]
    //
    // Set the padding value to be the most negative value that can be
    // represented by the datatype to ensure that any padding values will be equal
    // to or smaller than the actual maximum in the KH x KW patch.
    ETensor2<InEigenType> input_extract_patches =
        input_padded
            .extract_image_patches(kernel_h, kernel_w, stride_h, stride_w, 1, 1, Eigen::PADDING_VALID,
                                   std::numeric_limits<InEigenType>::lowest())
            .shuffle(Eigen::array<Eigen::Index, 5>{ 1, 2, 0, 3, 4 })
            .reshape(im2col_input_dims);

    // Get the maximum of the KHxHW patches along axis 0
    Eigen::Tensor<DenseIndex, 1> tensor_argmax = input_extract_patches.argmax(0);

    // 1D result with [N * H * W * C]
    ETensor1<OutEigenType> out_1d(this->out->getElementCount());

    // index input_patches with argmax array should give the result
    for (size_t i = 0; i < this->out->getElementCount(); i++)
    {
        out_1d(i) = (OutEigenType)input_extract_patches(tensor_argmax(i), i);
    }

    // reshape result to [N, H, W, C]
    this->out->getTensor() = out_1d.reshape(col2im_output_dims);

    return GraphNode::eval();
}

template <DType Dtype>
OpRFFT2d<Dtype>::OpRFFT2d(SubgraphTraverser* sgt_,
                          TosaAttributeBase* attribute_,
                          uint64_t id_)
    : GraphNode(sgt_, Op_RFFT2D, id_)
{
    setRequiredOperands(1, 2);
    setRequiredRank(3);
}

template <DType Dtype>
OpRFFT2d<Dtype>::~OpRFFT2d() {}


template <DType Dtype>
int OpRFFT2d<Dtype>::checkTensorAttributes()
{
    if (validateRequiredOperands())
        return 1;

    if (validateRequiredRank(inputs[0]) || validateRequiredRank(outputs[0]) ||
    validateRequiredRank(outputs[1]))
    {
        return 1;
    }

    if (inputs[0]->matchType(*outputs[0]) || inputs[0]->matchType(*outputs[1]))
    {
        printNodeValidationError("OpRFFT2d: input and output tensor type mismatch");
        return 1;
    }

    in  = dynamic_cast<TosaReference::TensorTemplate<TIn>*>(inputs[0]);
    out_real = dynamic_cast<TosaReference::TensorTemplate<TOut>*>(outputs[0]);
    out_imag = dynamic_cast<TosaReference::TensorTemplate<TOut>*>(outputs[1]);

    ASSERT_MEM(in && out_real && out_imag);

    auto is_power_of_two = [](int32_t n) -> bool
    {
        return (n & (n-1)) == 0 && n > 0;
    };

    // Input shape: [N, H, W]
    if (!is_power_of_two(in->getShape()[1]) || !is_power_of_two(in->getShape()[2]))
    {
        printNodeValidationError("OpRFFT2d: input height and width must be a power of two");
        return 1;
    }

    // Output shape: [N, H, W / 2 + 1]
    bool output_check = true;
    for (int32_t i = 0; i < out_real->getRank(); i++)
    {
        if (out_real->getShape()[i] != out_imag->getShape()[i])
        {
            output_check = false;
            break;
        }
    }
    if (!output_check)
    {
        printNodeValidationError(
            "OpRFFT2d: Mismatch between real output shape and imaginary output shape");
        return 1;
    }

    if (in->getShape()[0] != out_real->getShape()[0]) {
        printNodeValidationError("OpRFFT2d: input and output batch size don't match");
        return 1;
    }
    if (in->getShape()[1] != out_real->getShape()[1]) {
        printNodeValidationError("OpRFFT2d: input and output height don't match");
        return 1;
    }
    if (in->getShape()[2] / 2 + 1 != out_real->getShape()[2]) {
        printNodeValidationError("OpRFFT2d:  output width is expected to match input width / 2 + 1");
        return 1;
    }

    return 0;
}

template <DType Dtype>
int OpRFFT2d<Dtype>::eval()
{
    int32_t in_batch = in->getShape()[0];
    int32_t in_height = in->getShape()[1];
    int32_t in_width = in->getShape()[2];

    int32_t out_real_batch = out_real->getShape()[0];
    int32_t out_real_height = out_real->getShape()[1];
    int32_t out_real_width = out_real->getShape()[2];

    int32_t out_imag_batch = out_imag->getShape()[0];
    int32_t out_imag_height = out_imag->getShape()[1];
    int32_t out_imag_width = out_imag->getShape()[2];

    DEBUG_INFO(OP,
               "perform OpRFFT2d, input.shape=[%d,%d,%d], output_real.shape=[%d,%d,%d], "
               "output_imag.shape=[%d,%d,%d]",
               in_batch, in_height, in_width,
               out_real_batch, out_real_height, out_real_width,
               out_imag_batch, out_imag_height, out_imag_width);

    OutEigenType sum_real, sum_imag, a;

    for (int n = 0; n < in_batch; n++)
    {
        for (int oy = 0; oy < out_real_height; oy++)
        {
            for (int ox = 0; ox < out_real_width; ox++)
            {
                sum_real = 0.0;
                sum_imag = 0.0;
                for (int iy = 0; iy < in_height; iy++)
                {
                    for (int ix = 0; ix < in_width; ix++)
                    {
                        // Use explicit cast to ensure intermmediate calculations are completed using OutEigenType
                        a = 2 * M_PI * ((iy * (OutEigenType)oy) / in_height + (ix * (OutEigenType)ox) / in_width);
                        sum_real += this->in->getTensor()(n, iy, ix) * cos(a);
                        sum_imag += -this->in->getTensor()(n, iy, ix) * sin(a);
                    }
                }
                this->out_real->getTensor()(n, oy, ox) = sum_real;
                this->out_imag->getTensor()(n, oy, ox) = sum_imag;
            }
        }
    }

    return GraphNode::eval();
}

template <DType InDtype, DType WeightDtype, DType OutDtype>
OpTransposeConv2d<InDtype, WeightDtype, OutDtype>::OpTransposeConv2d(SubgraphTraverser* sgt_,
                                                           TosaAttributeBase* attribute_,
                                                           uint64_t id_)
    : GraphNode(sgt_, Op_TRANSPOSE_CONV2D, id_)
{
    setRequiredOperands(3, 1);
    setRequiredRank(4);

    INIT_ATTRIBUTE(TransposeConv);
}

template <DType InDtype, DType WeightDtype, DType OutDtype>
OpTransposeConv2d<InDtype, WeightDtype, OutDtype>::~OpTransposeConv2d()
{
    if (attribute)
        delete attribute;
}

template <DType InDtype, DType WeightDtype, DType OutDtype>
int OpTransposeConv2d<InDtype, WeightDtype, OutDtype>::checkTensorAttributes()
{
    if (validateRequiredOperands())
        return 1;

    if (validateRequiredRank(inputs[0]) || validateRequiredRank(inputs[1]) || validateRequiredRank(outputs[0]))
    {
        return 1;
    }

    ERROR_IF(outputs[0]->getDtype() != OutDtype,
                "OpTransposeConv2d: Output data type not supported for this configuration of operator");

    input  = dynamic_cast<TosaReference::TensorTemplate<TIn>*>(inputs[0]);
    weight = dynamic_cast<TosaReference::TensorTemplate<TWeight>*>(inputs[1]);
    bias   = dynamic_cast<TosaReference::TensorTemplate<TBias>*>(inputs[2]);
    output = dynamic_cast<TosaReference::TensorTemplate<TOut>*>(outputs[0]);

    if (attribute->out_pad().size() != 4)
    {
        printNodeValidationError("OpTransposeConv2d: illegal size for attribute out_pad");
        return 1;
    }

    if (attribute->stride().size() != 2)
    {
        printNodeValidationError("OpTransposeConv2d: illegal size for attribute stride");
        return 1;
    }

    if (attribute->output_shape().size() != 4)
    {
        printNodeValidationError("OpTransposeConv2d: illegal size for attribute output_shape");
        return 1;
    }



    for (int32_t i : attribute->stride())
    {
        if (i < 1)
        {
            printNodeValidationError("OpTransposeConv2d: At least one stride is smaller than one");
            return 1;
        }
    }

    for (int d = 0; d < 4; d++)
    {
        if (attribute->output_shape()[d] != this->output->getShape()[d])
        {
            printNodeValidationError("OpTransposeConv2d: illegal size for attribute output_shape");
            return 1;
        }
    }

    int32_t IH = input->getShape()[1];
    int32_t IW = input->getShape()[2];
    int32_t OH = output->getShape()[1];
    int32_t OW = output->getShape()[2];

    int32_t stride_y = attribute->stride()[0];
    int32_t stride_x = attribute->stride()[1];
    int32_t kernel_h = weight->getShape()[1];
    int32_t kernel_w = weight->getShape()[2];

    int32_t out_pad_top    = attribute->out_pad()[0];
    int32_t out_pad_bottom = attribute->out_pad()[1];
    int32_t out_pad_left   = attribute->out_pad()[2];
    int32_t out_pad_right  = attribute->out_pad()[3];

    for (size_t i = 0; i < attribute->out_pad().size(); i++)
    {
        ERROR_IF(attribute->out_pad()[i] <= -(weight->getShape()[(i / 2) + 1]), "OpTransposeConv2d: At least one out_pad value is larger than kernel size");
    }

    int32_t H = (IH - 1) * stride_y + out_pad_top + out_pad_bottom + kernel_h;
    int32_t W = (IW - 1) * stride_x + out_pad_left + out_pad_right + kernel_w;

    if ((OH != H) || (OW != W))
    {
        std::string msg = "OpTransposeConv2d: Mismatch between output shape provided and expected output shape (" +
            std::to_string(H) + "," +
            std::to_string(W) + ")";
        printNodeValidationError(msg.c_str());
        return 1;
    }

    ERROR_IF(InDtype != DType_INT8 && attribute->input_zp() != 0, "OpTransposeConv2d: Input zeropoint must be zero for non int8_t data");
    ERROR_IF(WeightDtype != DType_INT8 && attribute->weight_zp() != 0, "OpTransposeConv2d: Weight zeropoint must be zero for non int8_t data");

    return 0;
}

template <DType InDtype, DType WeightDtype, DType OutDtype>
int OpTransposeConv2d<InDtype, WeightDtype, OutDtype>::eval()
{
    int in_batch    = this->input->getShape()[0];
    int in_height   = this->input->getShape()[1];
    int in_width    = this->input->getShape()[2];
    int in_channels = this->input->getShape()[3];

    int f_out_channels = this->weight->getShape()[0];
    int f_height       = this->weight->getShape()[1];
    int f_width        = this->weight->getShape()[2];
    int f_in_channels  = this->weight->getShape()[3];

    int b_out_channels = this->bias->getShape()[0];

    int out_batch    = this->output->getShape()[0];
    int out_height   = this->output->getShape()[1];
    int out_width    = this->output->getShape()[2];
    int out_channels = this->output->getShape()[3];

    int out_pad_top    = this->attribute->out_pad()[0];
    int out_pad_bottom = this->attribute->out_pad()[1];
    int out_pad_left   = this->attribute->out_pad()[2];
    int out_pad_right  = this->attribute->out_pad()[3];

    int stride_h = this->attribute->stride()[0];
    int stride_w = this->attribute->stride()[1];

    ERROR_IF(in_batch != out_batch, "OpTransposeConv2d: tensor batch mismatch %d != %d", in_batch, out_batch);
    ERROR_IF(f_in_channels != in_channels, "OpTransposeConv2d: tensor input channel mismatch %d != %d", f_in_channels,
             in_channels);
    ERROR_IF(f_out_channels != out_channels, "OpTransposeConv2d: tensor output channel mismatch %d != %d",
             f_out_channels, out_channels);
    ERROR_IF(b_out_channels != out_channels, "OpDepthwiseConv2d: bias channels mismatch %d != %d", b_out_channels,
             out_channels);

    DEBUG_INFO(OP,
               "perform OpTransposeConv2d, input.shape=[%d,%d,%d,%d], weight.shape=[%d,%d,%d,%d], "
               "output.shape=[%d,%d,%d,%d], stride=[%d,%d], out_pad=[%d,%d,%d,%d]",
               in_batch, in_height, in_width, in_channels, f_height, f_width, f_out_channels, f_in_channels,
               out_batch, out_height, out_width, out_channels, stride_h, stride_w, out_pad_top,
               out_pad_bottom, out_pad_left, out_pad_right);

    TIn input_val      = this->input->getTensor();
    TWeight weight_val = this->weight->getTensor();
    if (InDtype == DType_INT8 || WeightDtype == DType_INT8)
    {
        input_val  = input_val - (InEigenType)attribute->input_zp();
        weight_val = weight_val - (WeightEigenType)attribute->weight_zp();
    }

    Eigen::array<Eigen::Index, 4> reshape_dim;
    reshape_dim.fill(1);
    reshape_dim[3] = b_out_channels;

    Eigen::array<Eigen::Index, 4> bcast;
    bcast[0] = out_batch;
    bcast[1] = out_height;
    bcast[2] = out_width;
    bcast[3] = 1;

    // initialize with bias
    this->output->getTensor() = this->bias->getTensor().reshape(reshape_dim).broadcast(bcast);

    int out_x_origin, out_y_origin;
    int out_x, out_y;

    // reference implementation from: tensorflow/tensorflow/lite/kernels/internal/reference/reference_ops.h
    for (int ob = 0; ob < out_batch; ob++)
    {
        for (int ih = 0; ih < in_height; ih++)
        {
            for (int iw = 0; iw < in_width; iw++)
            {
                out_x_origin = iw * stride_w + out_pad_left;
                out_y_origin = ih * stride_h + out_pad_top;
                for (int ic = 0; ic < in_channels; ic++)
                {
                    for (int fh = 0; fh < f_height; fh++)
                    {
                        for (int fw = 0; fw < f_width; fw++)
                        {
                            out_x = out_x_origin + fw;
                            out_y = out_y_origin + fh;
                            for (int oc = 0; oc < out_channels; oc++)
                            {
                                if ((out_x >= 0 && out_x < out_width) && (out_y >= 0 && out_y < out_height))
                                {
                                    this->output->getTensor()(ob, out_y, out_x, oc) +=
                                        (OutEigenType) ((AccEigenType)input_val(ob, ih, iw, ic) *
                                            (AccEigenType)weight_val(oc, fh, fw, ic));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if (OutDtype == DType_INT48)
    {
        this->output->getTensor() = this->output->getTensor().cwiseMax((OutEigenType)AccQMin);
        this->output->getTensor() = this->output->getTensor().cwiseMin((OutEigenType)AccQMax);
    }

    return GraphNode::eval();
}

// template explicit instantiation
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpArgMax, FP16);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpArgMax, BF16);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpArgMax, FP32);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpArgMax, INT8);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpArgMax, INT16);

DEF_INSTANTIATE_TWO_TYPE(OpAvgPool2d, FP16, FP16);
DEF_INSTANTIATE_TWO_TYPE(OpAvgPool2d, FP16, FP32);
DEF_INSTANTIATE_TWO_TYPE(OpAvgPool2d, BF16, FP32);
DEF_INSTANTIATE_TWO_TYPE(OpAvgPool2d, FP32, FP32);
DEF_INSTANTIATE_TWO_TYPE(OpAvgPool2d, INT8, INT32);
DEF_INSTANTIATE_TWO_TYPE(OpAvgPool2d, INT16, INT32);

                                // [in_t, weight_t, out_t]
DEF_INSTANTIATE_THREE_TYPE(OpConv2d, FP16, FP16, FP16);
DEF_INSTANTIATE_THREE_TYPE(OpConv2d, FP16, FP16, FP32);
DEF_INSTANTIATE_THREE_TYPE(OpConv2d, BF16, BF16, FP32);
DEF_INSTANTIATE_THREE_TYPE(OpConv2d, FP32, FP32, FP32);
DEF_INSTANTIATE_THREE_TYPE(OpConv2d, INT8, INT4, INT32);
DEF_INSTANTIATE_THREE_TYPE(OpConv2d, INT8, INT8, INT32);
DEF_INSTANTIATE_THREE_TYPE(OpConv2d, INT16, INT8, INT48);

DEF_INSTANTIATE_THREE_TYPE(OpConv3d, FP16, FP16, FP16);
DEF_INSTANTIATE_THREE_TYPE(OpConv3d, FP16, FP16, FP32);
DEF_INSTANTIATE_THREE_TYPE(OpConv3d, BF16, BF16, FP32);
DEF_INSTANTIATE_THREE_TYPE(OpConv3d, FP32, FP32, FP32);
DEF_INSTANTIATE_THREE_TYPE(OpConv3d, INT8, INT4, INT32);
DEF_INSTANTIATE_THREE_TYPE(OpConv3d, INT8, INT8, INT32);
DEF_INSTANTIATE_THREE_TYPE(OpConv3d, INT16, INT8, INT48);

DEF_INSTANTIATE_THREE_TYPE(OpDepthwiseConv2d, FP16, FP16, FP16);
DEF_INSTANTIATE_THREE_TYPE(OpDepthwiseConv2d, FP16, FP16, FP32);
DEF_INSTANTIATE_THREE_TYPE(OpDepthwiseConv2d, BF16, BF16, FP32);
DEF_INSTANTIATE_THREE_TYPE(OpDepthwiseConv2d, FP32, FP32, FP32);
DEF_INSTANTIATE_THREE_TYPE(OpDepthwiseConv2d, INT8, INT4, INT32);
DEF_INSTANTIATE_THREE_TYPE(OpDepthwiseConv2d, INT8, INT8, INT32);
DEF_INSTANTIATE_THREE_TYPE(OpDepthwiseConv2d, INT16, INT8, INT48);

DEF_INSTANTIATE_THREE_TYPE(OpFullyConnected, FP16, FP16, FP16);
DEF_INSTANTIATE_THREE_TYPE(OpFullyConnected, FP16, FP16, FP32);
DEF_INSTANTIATE_THREE_TYPE(OpFullyConnected, BF16, BF16, FP32);
DEF_INSTANTIATE_THREE_TYPE(OpFullyConnected, FP32, FP32, FP32);
DEF_INSTANTIATE_THREE_TYPE(OpFullyConnected, INT8, INT4, INT32);
DEF_INSTANTIATE_THREE_TYPE(OpFullyConnected, INT8, INT8, INT32);
DEF_INSTANTIATE_THREE_TYPE(OpFullyConnected, INT16, INT8, INT48);

DEF_INSTANTIATE_TWO_TYPE(OpMatMul, INT8, INT32);
DEF_INSTANTIATE_TWO_TYPE(OpMatMul, INT16, INT48);
DEF_INSTANTIATE_TWO_TYPE(OpMatMul, FP16, FP16);
DEF_INSTANTIATE_TWO_TYPE(OpMatMul, FP16, FP32);
DEF_INSTANTIATE_TWO_TYPE(OpMatMul, BF16, FP32);
DEF_INSTANTIATE_TWO_TYPE(OpMatMul, FP32, FP32);

DEF_INSTANTIATE_ONE_TYPE(OpMaxPool2d, FP16);
DEF_INSTANTIATE_ONE_TYPE(OpMaxPool2d, BF16);
DEF_INSTANTIATE_ONE_TYPE(OpMaxPool2d, FP32);
DEF_INSTANTIATE_ONE_TYPE(OpMaxPool2d, INT8);
DEF_INSTANTIATE_ONE_TYPE(OpMaxPool2d, INT16);

DEF_INSTANTIATE_ONE_TYPE(OpRFFT2d, FP32);

DEF_INSTANTIATE_THREE_TYPE(OpTransposeConv2d, FP16, FP16, FP16);
DEF_INSTANTIATE_THREE_TYPE(OpTransposeConv2d, FP16, FP16, FP32);
DEF_INSTANTIATE_THREE_TYPE(OpTransposeConv2d, BF16, BF16, FP32);
DEF_INSTANTIATE_THREE_TYPE(OpTransposeConv2d, FP32, FP32, FP32);
DEF_INSTANTIATE_THREE_TYPE(OpTransposeConv2d, INT8, INT4, INT32);
DEF_INSTANTIATE_THREE_TYPE(OpTransposeConv2d, INT8, INT8, INT32);
DEF_INSTANTIATE_THREE_TYPE(OpTransposeConv2d, INT16, INT8, INT48);

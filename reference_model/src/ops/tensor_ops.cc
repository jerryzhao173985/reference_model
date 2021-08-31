
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

#include "tensor_ops.h"
#include "quant_util.h"
#include "template_types.h"

using namespace TosaReference;
using namespace Eigen;
using namespace tosa;

template <int Rank, DType Dtype>
OpArgMax<Rank, Dtype>::OpArgMax(SubgraphTraverser* sgt_,
                                TosaAttributeBase* attribute_,
                                TosaQuantInfoBase* qinfo_,
                                uint64_t id_)
    : GraphNode(sgt_, Op_ARGMAX, id_)
{
    setRequiredOperands(1, 1);
    setRequiredRank(0, 6);

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

    if (validateRequiredRank(inputs[0]) || validateRequiredRank(outputs[0]))
    {
        return 1;
    }

    input  = dynamic_cast<TosaReference::TensorTemplate<TIn>*>(inputs[0]);
    output = dynamic_cast<TosaReference::TensorTemplate<TOut>*>(outputs[0]);

    return 0;
}

template <int Rank, DType Dtype>
int OpArgMax<Rank, Dtype>::eval()
{
    Eigen::Tensor<DenseIndex, Rank - 1> index = this->input->getTensor().argmax(attribute->axis());

    this->output->getTensor() = index.unaryExpr([](DenseIndex in) -> OutEigenType { return (OutEigenType)in; });

    return GraphNode::eval();
}

template <DType Dtype>
OpAvgPool2d<Dtype>::OpAvgPool2d(SubgraphTraverser* sgt_,
                                TosaAttributeBase* attribute_,
                                TosaQuantInfoBase* qinfo_,
                                uint64_t id_)
    : GraphNode(sgt_, Op_AVG_POOL2D, id_)
{
    setRequiredOperands(1, 1);
    setRequiredRank(4);

    INIT_ATTRIBUTE(Pool);
    INIT_QINFO(Unary);
}

template <DType Dtype>
OpAvgPool2d<Dtype>::~OpAvgPool2d()
{
    if (attribute)
        delete attribute;
}

template <DType Dtype>
int OpAvgPool2d<Dtype>::checkTensorAttributes()
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

    if (attribute->padding().size() != 4)
    {
        printNodeValidationError("OpAvgPool2d: illegal size for attribute padding");
        return 1;
    }

    if (attribute->kernel().size() != 2)
    {
        printNodeValidationError("OpAvgPool2d: illegal size for attribute kernel");
        return 1;
    }

    if (attribute->stride().size() != 2)
    {
        printNodeValidationError("OpAvgPool2d: illegal size for attribute stride");
        return 1;
    }

    return 0;
}

template <DType Dtype>
ETensor1<int32_t> OpAvgPool2d<Dtype>::calculate_div_map_1d(int in_size, int out_size, int kernel_size, int stride)
{
    ETensor1<int32_t> result(out_size);

    int32_t total_pad = (out_size - 1) * stride + kernel_size - in_size;
    total_pad         = total_pad < 0 ? 0 : total_pad;

    int32_t pad_left  = total_pad >> 1;
    int32_t pad_right = total_pad - pad_left;

    result.setConstant(kernel_size);

    // the index left to 'left_index' and index right to 'right_index' indicates
    // the input window of this output covers a pad bit
    int32_t left_index  = pad_left / stride;
    int32_t right_index = pad_right / stride;

    // minus the number of pad bit this index cover
    while (left_index >= 0)
    {
        result(left_index) -= (pad_left - left_index * stride);
        left_index--;
    }

    while (right_index >= 0)
    {
        result(out_size - 1 - right_index) -= (pad_right - right_index * stride);
        right_index--;
    }

    return result;
}

// assuming input and output tensor have same scales like tflite reference
// so no need to scale input and output
template <DType Dtype>
int OpAvgPool2d<Dtype>::eval()
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

    int padding_top    = this->attribute->padding()[0];
    int padding_bottom = this->attribute->padding()[1];
    int padding_left   = this->attribute->padding()[2];
    int padding_right  = this->attribute->padding()[3];
    int kernel_h       = this->attribute->kernel()[0];
    int kernel_w       = this->attribute->kernel()[1];
    int stride_h       = this->attribute->stride()[0];
    int stride_w       = this->attribute->stride()[1];

    DEBUG_INFO(OP,
               "perform AvgPool2d, input.shape=[%d,%d,%d,%d], output.shape=[%d,%d,%d,%d], kernel=[%d,%d], "
               "stride=[%d,%d], padding=[%d,%d,%d,%d]",
               in_batch, in_height, in_width, in_channels, out_batch, out_height, out_width, out_channels, kernel_h,
               kernel_w, stride_h, stride_w, padding_top, padding_bottom, padding_left, padding_right);

    Eigen::array<Eigen::Index, 2> im2col_input_dims;
    im2col_input_dims[0] = kernel_h * kernel_w;
    im2col_input_dims[1] = out_batch * out_height * out_width * out_channels;

    Eigen::array<Eigen::Index, 4> col2im_output_dims;
    col2im_output_dims[0] = out_batch;
    col2im_output_dims[1] = out_height;
    col2im_output_dims[2] = out_width;
    col2im_output_dims[3] = out_channels;

    Eigen::array<std::pair<int32_t, int32_t>, 4> padding;
    padding[0] = std::make_pair(0, 0);
    padding[1] = std::make_pair(padding_top, padding_bottom);
    padding[2] = std::make_pair(padding_left, padding_right);
    padding[3] = std::make_pair(0, 0);

    ETensor4<InEigenType> input_val = this->in->getTensor();
    if (this->qinfo)
    {
        input_val = input_val - (InEigenType)this->qinfo->input_zp();
    }

    ETensor4<InEigenType> input_padded = input_val.pad(padding);

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
    ETensor1<int32_t> div_map_h = calculate_div_map_1d(in_height, out_height, kernel_h, stride_h);
    ETensor1<int32_t> div_map_w = calculate_div_map_1d(in_width, out_width, kernel_w, stride_w);
    Eigen::array<Eigen::IndexPair<Eigen::Index>, 1> contract_dims = { Eigen::IndexPair<Eigen::Index>(1, 0) };
    Eigen::array<Eigen::Index, 4> bcast{ out_batch, 1, 1, out_channels };

    ETensor4<int32_t> div_map =
        div_map_h.reshape(Eigen::array<Eigen::Index, 2>{ out_height, 1 })
            .contract(div_map_w.reshape(Eigen::array<Eigen::Index, 2>{ 1, out_width }), contract_dims)
            .reshape(Eigen::array<Eigen::Index, 4>{ 1, out_height, out_width, 1 })
            .broadcast(bcast);

    if (Dtype != DType_FLOAT)
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
        this->out->getTensor() = this->out->getTensor() + (OutEigenType)(this->qinfo->output_zp());
        this->out->getTensor() = this->out->getTensor().cwiseMax((OutEigenType)QMin);
        this->out->getTensor() = this->out->getTensor().cwiseMin((OutEigenType)QMax);
    }
    else
    {
        this->out->getTensor() = (sum / div_map.template cast<AccEigenType>()).template cast<OutEigenType>();
    }

    return GraphNode::eval();
}

template <DType InDtype, DType WeightDtype>
OpConv2d<InDtype, WeightDtype>::OpConv2d(SubgraphTraverser* sgt_,
                                         TosaAttributeBase* attribute_,
                                         TosaQuantInfoBase* qinfo_,
                                         uint64_t id_)
    : GraphNode(sgt_, Op_CONV2D, id_)
{
    setRequiredOperands(3, 1);
    setRequiredRank(4);

    INIT_ATTRIBUTE(Conv);
    INIT_QINFO(Conv);
}

template <DType InDtype, DType WeightDtype>
OpConv2d<InDtype, WeightDtype>::~OpConv2d()
{
    if (attribute)
        delete attribute;
    if (qinfo)
        delete qinfo;
}

template <DType InDtype, DType WeightDtype>
int OpConv2d<InDtype, WeightDtype>::checkTensorAttributes()
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

    input  = dynamic_cast<TosaReference::TensorTemplate<TIn>*>(inputs[0]);
    weight = dynamic_cast<TosaReference::TensorTemplate<TWeight>*>(inputs[1]);
    bias   = dynamic_cast<TosaReference::TensorTemplate<TBias>*>(inputs[2]);
    output = dynamic_cast<TosaReference::TensorTemplate<TAcc>*>(outputs[0]);

    if (attribute->padding().size() != 4)
    {
        printNodeValidationError("OpConv2d: illegal size for attribute padding");
        return 1;
    }

    if (attribute->stride().size() != 2)
    {
        printNodeValidationError("OpConv2d: illegal size for attribute stride");
        return 1;
    }

    if (attribute->dilation().size() != 2)
    {
        printNodeValidationError("OpConv2d: illegal size for attribute dilation");
        return 1;
    }

    return 0;
}

template <DType InDtype, DType WeightDtype>
int OpConv2d<InDtype, WeightDtype>::eval()
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

    int padding_top    = this->attribute->padding()[0];
    int padding_bottom = this->attribute->padding()[1];
    int padding_left   = this->attribute->padding()[2];
    int padding_right  = this->attribute->padding()[3];
    int stride_h       = this->attribute->stride()[0];
    int stride_w       = this->attribute->stride()[1];
    int dilation_h     = this->attribute->dilation()[0];
    int dilation_w     = this->attribute->dilation()[1];

    DEBUG_INFO(OP,
               "perform OpConv2d, input.shape=[%d,%d,%d,%d], weight.shape=[%d,%d,%d,%d], output.shape=[%d,%d,%d,%d], "
               "stride=[%d,%d], dilation=[%d,%d], padding=[%d,%d,%d,%d]",
               in_batch, in_height, in_width, in_channels, f_height, f_width, f_in_channels, f_out_channels, out_batch,
               out_height, out_width, out_channels, stride_h, stride_w, dilation_h, dilation_w, padding_top,
               padding_bottom, padding_left, padding_right);

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

    Eigen::array<std::pair<int32_t, int32_t>, 4> padding;
    padding[0] = std::make_pair(0, 0);
    padding[1] = std::make_pair(padding_top, padding_bottom);
    padding[2] = std::make_pair(padding_left, padding_right);
    padding[3] = std::make_pair(0, 0);

    TIn input_val      = this->input->getTensor();
    TWeight weight_val = this->weight->getTensor();
    if (this->qinfo)
    {
        input_val  = input_val - (InEigenType)this->qinfo->input_zp();
        weight_val = weight_val - (WeightEigenType)this->qinfo->weight_zp();
    }

    ETensor4<InEigenType> input_padded = input_val.pad(padding);

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
    ETensor2<AccEigenType> bias_2d = this->bias->getTensor().reshape(bias_reshaped_dims).broadcast(bias_bcast_dims);

    // output matrix is [N * H * W, C]
    ETensor2<AccEigenType> contracted_result =
        im2col_input.template cast<AccEigenType>().contract(im2col_weight.template cast<AccEigenType>(), contract_dims);

    // adding bias
    ETensor2<AccEigenType> biased_output = contracted_result + bias_2d.template cast<AccEigenType>();

    // reshape back to [N, H, W, C]
    this->output->getTensor() = biased_output.reshape(col2im_output_dims);

    if (AccDtype == DType_INT48)
    {
        this->output->getTensor() = this->output->getTensor().cwiseMax((AccEigenType)AccQMin);
        this->output->getTensor() = this->output->getTensor().cwiseMin((AccEigenType)AccQMax);
    }

    return GraphNode::eval();
}

template <DType InDtype, DType WeightDtype>
OpDepthwiseConv2d<InDtype, WeightDtype>::OpDepthwiseConv2d(SubgraphTraverser* sgt_,
                                                           TosaAttributeBase* attribute_,
                                                           TosaQuantInfoBase* qinfo_,
                                                           uint64_t id_)
    : GraphNode(sgt_, Op_DEPTHWISE_CONV2D, id_)
{
    setRequiredOperands(3, 1);
    setRequiredRank(4);

    INIT_ATTRIBUTE(Conv);
    INIT_QINFO(Conv);
}

template <DType InDtype, DType WeightDtype>
OpDepthwiseConv2d<InDtype, WeightDtype>::~OpDepthwiseConv2d()
{
    if (attribute)
        delete attribute;
    if (qinfo)
        delete qinfo;
}

template <DType InDtype, DType WeightDtype>
int OpDepthwiseConv2d<InDtype, WeightDtype>::checkTensorAttributes()
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

    input  = dynamic_cast<TosaReference::TensorTemplate<TIn>*>(inputs[0]);
    weight = dynamic_cast<TosaReference::TensorTemplate<TWeight>*>(inputs[1]);
    bias   = dynamic_cast<TosaReference::TensorTemplate<TBias>*>(inputs[2]);
    output = dynamic_cast<TosaReference::TensorTemplate<TAcc>*>(outputs[0]);

    if (attribute->padding().size() != 4)
    {
        printNodeValidationError("OpDepthwiseConv2d: illegal size for attribute padding");
        return 1;
    }

    if (attribute->stride().size() != 2)
    {
        printNodeValidationError("OpDepthwiseConv2d: illegal size for attribute stride");
        return 1;
    }

    if (attribute->dilation().size() != 2)
    {
        printNodeValidationError("OpDepthwiseConv2d: illegal size for attribute dilation");
        return 1;
    }

    return 0;
}

template <DType InDtype, DType WeightDtype>
int OpDepthwiseConv2d<InDtype, WeightDtype>::eval()
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

    int padding_top    = this->attribute->padding()[0];
    int padding_bottom = this->attribute->padding()[1];
    int padding_left   = this->attribute->padding()[2];
    int padding_right  = this->attribute->padding()[3];
    int stride_h       = this->attribute->stride()[0];
    int stride_w       = this->attribute->stride()[1];
    int dilation_h     = this->attribute->dilation()[0];
    int dilation_w     = this->attribute->dilation()[1];

    DEBUG_INFO(OP,
               "perform OpDepthwiseConv2d, input.shape=[%d,%d,%d,%d], weight.shape=[%d,%d,%d,%d], "
               "output.shape=[%d,%d,%d,%d], stride=[%d,%d], dilation=[%d,%d], padding=[%d,%d,%d,%d]",
               in_batch, in_height, in_width, in_channels, f_height, f_width, f_in_channels, f_multiplier, out_batch,
               out_height, out_width, out_channels, stride_h, stride_w, dilation_h, dilation_w, padding_top,
               padding_bottom, padding_left, padding_right);

    Eigen::array<std::pair<int32_t, int32_t>, 4> padding;
    padding[0] = std::make_pair(0, 0);
    padding[1] = std::make_pair(padding_top, padding_bottom);
    padding[2] = std::make_pair(padding_left, padding_right);
    padding[3] = std::make_pair(0, 0);

    TIn input_val      = this->input->getTensor();
    TWeight weight_val = this->weight->getTensor();
    if (this->qinfo)
    {
        input_val  = input_val - (InEigenType)this->qinfo->input_zp();
        weight_val = weight_val - (WeightEigenType)this->qinfo->weight_zp();
    }

    ETensor4<InEigenType> input_padded = input_val.pad(padding);

    // GEMM doesn't fit well with DepthwiseConv2d
    // 1. use extract_image_patches() to handle stride/dilation/padding
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
                                this->output->getTensor()(ob, oh, ow, ic * f_multiplier + cm) +=
                                    ((AccEigenType)input_extract_patches(ob, fh, fw, ow * out_height + oh, ic) *
                                     (AccEigenType)weight_val(fh, fw, ic, cm));
                            }
                        }
                    }
                }
            }
        }
    }

    if (AccDtype == DType_INT48)
    {
        this->output->getTensor() = this->output->getTensor().cwiseMax((AccEigenType)AccQMin);
        this->output->getTensor() = this->output->getTensor().cwiseMin((AccEigenType)AccQMax);
    }

    return GraphNode::eval();
}

template <DType InDtype, DType WeightDtype>
OpFullyConnected<InDtype, WeightDtype>::OpFullyConnected(SubgraphTraverser* sgt_,
                                                         TosaAttributeBase* attribute_,
                                                         TosaQuantInfoBase* qinfo_,
                                                         uint64_t id_)
    : GraphNode(sgt_, Op_FULLY_CONNECTED, id_)
{
    setRequiredOperands(3, 1);
    setRequiredRank(2);

    INIT_QINFO(Conv);
}

template <DType InDtype, DType WeightDtype>
OpFullyConnected<InDtype, WeightDtype>::~OpFullyConnected()
{
    if (qinfo)
        delete qinfo;
}

template <DType InDtype, DType WeightDtype>
int OpFullyConnected<InDtype, WeightDtype>::checkTensorAttributes()
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

    output = dynamic_cast<TosaReference::TensorTemplate<TAcc>*>(outputs[0]);

    return 0;
}

template <DType InDtype, DType WeightDtype>
int OpFullyConnected<InDtype, WeightDtype>::eval()
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
    if (this->qinfo)
    {
        input_val  = input_val - (InEigenType)this->qinfo->input_zp();
        weight_val = weight_val - (WeightEigenType)this->qinfo->weight_zp();
    }

    this->output->getTensor() =
        input_val.template cast<AccEigenType>().contract(weight_val.template cast<AccEigenType>(), dims) +
        this->bias->getTensor().reshape(bias_reshape).broadcast(bias_bcast);

    if (AccDtype == DType_INT48)
    {
        this->output->getTensor() = this->output->getTensor().cwiseMax((AccEigenType)AccQMin);
        this->output->getTensor() = this->output->getTensor().cwiseMin((AccEigenType)AccQMax);
    }
    return GraphNode::eval();
}

template <DType Dtype>
OpMatMul<Dtype>::OpMatMul(SubgraphTraverser* sgt_,
                          TosaAttributeBase* attribute_,
                          TosaQuantInfoBase* qinfo_,
                          uint64_t id_)
    : GraphNode(sgt_, Op_MATMUL, id_)
{
    setRequiredOperands(2, 1);
    setRequiredRank(3);

    INIT_QINFO(MatMul);
}

template <DType Dtype>
OpMatMul<Dtype>::~OpMatMul()
{
    if (qinfo)
        delete qinfo;
}

template <DType Dtype>
int OpMatMul<Dtype>::checkTensorAttributes()
{
    if (validateRequiredOperands())
        return 1;

    if (validateRequiredRank(inputs[0]) || validateRequiredRank(inputs[1]) || validateRequiredRank(outputs[0]))
    {
        return 1;
    }

    a      = dynamic_cast<TosaReference::TensorTemplate<TIn>*>(inputs[0]);
    b      = dynamic_cast<TosaReference::TensorTemplate<TIn>*>(inputs[1]);
    output = dynamic_cast<TosaReference::TensorTemplate<TAcc>*>(outputs[0]);

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

    return 0;
}

template <DType Dtype>
int OpMatMul<Dtype>::eval()
{
    typedef Eigen::Tensor<int, 1>::DimensionPair DimPair;
    Eigen::array<DimPair, 1> dims{ { DimPair(1, 0) } };

    TIn a_val = this->a->getTensor();
    TIn b_val = this->b->getTensor();
    if (this->qinfo)
    {
        a_val = a_val - (InEigenType)this->qinfo->a_zp();
        b_val = b_val - (InEigenType)this->qinfo->b_zp();
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
        TAcc output_rank3_val = output_rank2_val.reshape(output_rank3_shape);
        if (i == 0)
        {
            this->output->getTensor() = output_rank3_val;
        }
        else
        {
            TAcc temp                 = this->output->getTensor().concatenate(output_rank3_val, 0);
            this->output->getTensor() = temp;
        }
    }

    if (AccDtype == DType_INT48)
    {
        this->output->getTensor() = this->output->getTensor().cwiseMax((AccEigenType)AccQMin);
        this->output->getTensor() = this->output->getTensor().cwiseMin((AccEigenType)AccQMax);
    }

    return GraphNode::eval();
}

template <DType Dtype>
OpMaxPool2d<Dtype>::OpMaxPool2d(SubgraphTraverser* sgt_,
                                TosaAttributeBase* attribute_,
                                TosaQuantInfoBase* qinfo_,
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

    if (attribute->padding().size() != 4)
    {
        printNodeValidationError("OpMaxPool2d: illegal size for attribute padding");
        return 1;
    }

    if (attribute->kernel().size() != 2)
    {
        printNodeValidationError("OpMaxPool2d: illegal size for attribute kernel");
        return 1;
    }

    if (attribute->stride().size() != 2)
    {
        printNodeValidationError("OpMaxPool2d: illegal size for attribute stride");
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

    int padding_top    = this->attribute->padding()[0];
    int padding_bottom = this->attribute->padding()[1];
    int padding_left   = this->attribute->padding()[2];
    int padding_right  = this->attribute->padding()[3];
    int kernel_h       = this->attribute->kernel()[0];
    int kernel_w       = this->attribute->kernel()[1];
    int stride_h       = this->attribute->stride()[0];
    int stride_w       = this->attribute->stride()[1];

    DEBUG_INFO(OP,
               "perform MaxPool2d, input.shape=[%d,%d,%d,%d], output.shape=[%d,%d,%d,%d], kernel=[%d,%d], "
               "stride=[%d,%d], padding=[%d,%d,%d,%d]",
               in_batch, in_height, in_width, in_channels, out_batch, out_height, out_width, out_channels, kernel_h,
               kernel_w, stride_h, stride_w, padding_top, padding_bottom, padding_left, padding_right);

    Eigen::array<Eigen::Index, 2> im2col_input_dims;
    im2col_input_dims[0] = kernel_h * kernel_w;
    im2col_input_dims[1] = out_batch * out_height * out_width * out_channels;

    Eigen::array<Eigen::Index, 4> col2im_output_dims;
    col2im_output_dims[0] = out_batch;
    col2im_output_dims[1] = out_height;
    col2im_output_dims[2] = out_width;
    col2im_output_dims[3] = out_channels;

    Eigen::array<std::pair<int32_t, int32_t>, 4> padding;
    padding[0] = std::make_pair(0, 0);
    padding[1] = std::make_pair(padding_top, padding_bottom);
    padding[2] = std::make_pair(padding_left, padding_right);
    padding[3] = std::make_pair(0, 0);

    ETensor4<InEigenType> input_padded = this->in->getTensor().pad(padding, std::numeric_limits<InEigenType>::lowest());

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

template <DType InDtype, DType OutDtype>
OpTransposeConv2d<InDtype, OutDtype>::OpTransposeConv2d(SubgraphTraverser* sgt_,
                                                        TosaAttributeBase* attribute_,
                                                        TosaQuantInfoBase* qinfo_,
                                                        uint64_t id_)
    : GraphNode(sgt_, Op_TRANSPOSE_CONV2D, id_)
{
    setRequiredOperands(3, 1);
    setRequiredRank(4);

    INIT_ATTRIBUTE(TransposeConv);
    INIT_QINFO(Conv);
}

template <DType InDtype, DType OutDtype>
OpTransposeConv2d<InDtype, OutDtype>::~OpTransposeConv2d()
{
    if (attribute)
        delete attribute;
    if (qinfo)
        delete qinfo;
}

template <DType InDtype, DType OutDtype>
int OpTransposeConv2d<InDtype, OutDtype>::checkTensorAttributes()
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
    output = dynamic_cast<TosaReference::TensorTemplate<TAcc>*>(outputs[0]);

    if (attribute->outpad().size() != 2)
    {
        printNodeValidationError("OpTransposeConv2d: illegal size for attribute outpad");
        return 1;
    }

    if (attribute->stride().size() != 2)
    {
        printNodeValidationError("OpTransposeConv2d: illegal size for attribute stride");
        return 1;
    }

    if (attribute->dilation().size() != 2)
    {
        printNodeValidationError("OpTransposeConv2d: illegal size for attribute dilation");
        return 1;
    }

    if (attribute->output_shape().size() != 4)
    {
        printNodeValidationError("OpTransposeConv2d: illegal size for attribute output_shape");
        return 1;
    }

    for (int d = 0; d < 4; d++)
    {
        if (attribute->output_shape()[d] != this->output->getShape()[d])
        {
            printNodeValidationError("OpTransposeConv2d: illegal size for attribute output_shape");
            return 1;
        }
    }

    return 0;
}

template <DType InDtype, DType OutDtype>
int OpTransposeConv2d<InDtype, OutDtype>::eval()
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

    int padding_top  = this->attribute->outpad()[0];
    int padding_left = this->attribute->outpad()[1];
    int stride_h     = this->attribute->stride()[0];
    int stride_w     = this->attribute->stride()[1];
    int dilation_h   = this->attribute->dilation()[0];
    int dilation_w   = this->attribute->dilation()[1];

    ERROR_IF(in_batch != out_batch, "OpTransposeConv2d: tensor batch mismatch %d != %d", in_batch, out_batch);
    ERROR_IF(f_in_channels != in_channels, "OpTransposeConv2d: tensor input channel mismatch %d != %d", f_in_channels,
             in_channels);
    ERROR_IF(f_out_channels != out_channels, "OpTransposeConv2d: tensor output channel mismatch %d != %d",
             f_out_channels, out_channels);
    ERROR_IF(b_out_channels != out_channels, "OpDepthwiseConv2d: bias channels mismatch %d != %d", b_out_channels,
             out_channels);

    DEBUG_INFO(OP,
               "perform OpTransposeConv2d, input.shape=[%d,%d,%d,%d], weight.shape=[%d,%d,%d,%d], "
               "output.shape=[%d,%d,%d,%d], stride=[%d,%d], dilation=[%d,%d], padding=[%d,%d]",
               in_batch, in_height, in_width, in_channels, f_height, f_width, f_out_channels, f_in_channels, out_batch,
               out_height, out_width, out_channels, stride_h, stride_w, dilation_h, dilation_w, padding_top,
               padding_left);

    TIn input_val      = this->input->getTensor();
    TWeight weight_val = this->weight->getTensor();
    if (this->qinfo)
    {
        input_val  = input_val - (InEigenType)this->qinfo->input_zp();
        weight_val = weight_val - (WeightEigenType)this->qinfo->weight_zp();
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
                out_x_origin = iw * stride_w - padding_left;
                out_y_origin = ih * stride_h - padding_top;
                for (int ic = 0; ic < in_channels; ic++)
                {
                    for (int fh = 0; fh < f_height; fh++)
                    {
                        for (int fw = 0; fw < f_width; fw++)
                        {
                            out_x = out_x_origin + fw * dilation_w;
                            out_y = out_y_origin + fh * dilation_h;
                            for (int oc = 0; oc < out_channels; oc++)
                            {
                                if ((out_x >= 0 && out_x < out_width) && (out_y >= 0 && out_y < out_height))
                                {
                                    this->output->getTensor()(ob, out_y, out_x, oc) +=
                                        ((AccEigenType)input_val(ob, ih, iw, ic) *
                                         (AccEigenType)weight_val(oc, fh, fw, ic));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if (AccDtype == DType_INT48)
    {
        this->output->getTensor() = this->output->getTensor().cwiseMax((AccEigenType)AccQMin);
        this->output->getTensor() = this->output->getTensor().cwiseMin((AccEigenType)AccQMax);
    }

    return GraphNode::eval();
}

// template explicit instantiation
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpArgMax, FLOAT);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpArgMax, INT8);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpArgMax, INT16);

DEF_INSTANTIATE_ONE_TYPE(OpAvgPool2d, FLOAT)
DEF_INSTANTIATE_ONE_TYPE(OpAvgPool2d, INT8)
DEF_INSTANTIATE_ONE_TYPE(OpAvgPool2d, INT16)

DEF_INSTANTIATE_TWO_TYPE(OpConv2d, FLOAT, FLOAT);
DEF_INSTANTIATE_TWO_TYPE(OpConv2d, INT8, INT4);
DEF_INSTANTIATE_TWO_TYPE(OpConv2d, INT8, INT8);
DEF_INSTANTIATE_TWO_TYPE(OpConv2d, INT16, INT8);

DEF_INSTANTIATE_TWO_TYPE(OpDepthwiseConv2d, FLOAT, FLOAT);
DEF_INSTANTIATE_TWO_TYPE(OpDepthwiseConv2d, INT8, INT4);
DEF_INSTANTIATE_TWO_TYPE(OpDepthwiseConv2d, INT8, INT8);
DEF_INSTANTIATE_TWO_TYPE(OpDepthwiseConv2d, INT16, INT8);

DEF_INSTANTIATE_TWO_TYPE(OpFullyConnected, FLOAT, FLOAT);
DEF_INSTANTIATE_TWO_TYPE(OpFullyConnected, INT8, INT4);
DEF_INSTANTIATE_TWO_TYPE(OpFullyConnected, INT8, INT8);
DEF_INSTANTIATE_TWO_TYPE(OpFullyConnected, INT16, INT8);

DEF_INSTANTIATE_ONE_TYPE(OpMatMul, INT8);
DEF_INSTANTIATE_ONE_TYPE(OpMatMul, INT16);
DEF_INSTANTIATE_ONE_TYPE(OpMatMul, FLOAT);

DEF_INSTANTIATE_ONE_TYPE(OpMaxPool2d, FLOAT);
DEF_INSTANTIATE_ONE_TYPE(OpMaxPool2d, INT8);
DEF_INSTANTIATE_ONE_TYPE(OpMaxPool2d, INT16);

DEF_INSTANTIATE_TWO_TYPE(OpTransposeConv2d, FLOAT, FLOAT);
DEF_INSTANTIATE_TWO_TYPE(OpTransposeConv2d, INT8, INT4);
DEF_INSTANTIATE_TWO_TYPE(OpTransposeConv2d, INT8, INT8);
DEF_INSTANTIATE_TWO_TYPE(OpTransposeConv2d, INT16, INT8);

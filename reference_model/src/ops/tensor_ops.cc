
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

#include "tensor_ops.h"
#include "dtype_limits.h"
#include "half.hpp"
#include "quant_util.h"
#include "template_types.h"

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

    if ((full_H % stride_y != 0) || (full_W % stride_x != 0))
    {
        msg = "Parameters must yield exact integer output dimensions";
        return 1;
    }

    if ((OH != (full_H / stride_y) + 1) || (OW != (full_W / stride_x) + 1))
    {
        msg = "Mismatch between output shape provided and expected output shape (" +
              std::to_string((full_H / stride_y) + 1) + "," + std::to_string((full_W / stride_x) + 1) + ")";
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
                         TOSA_REF_TYPE InDtype,
                         TOSA_REF_TYPE WeightDtype,
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
    int32_t ID       = conv_dimension == 3 ? input_shape[1] : 1;
    int32_t IH       = input_shape[1 + offset_d];
    int32_t IW       = input_shape[2 + offset_d];
    int32_t OD       = conv_dimension == 3 ? output_shape[1] : 1;
    int32_t OH       = output_shape[1 + offset_d];
    int32_t OW       = output_shape[2 + offset_d];

    int32_t stride_d   = conv_dimension == 3 ? attribute->stride()[0] : 1;
    int32_t stride_y   = attribute->stride()[0 + offset_d];
    int32_t stride_x   = attribute->stride()[1 + offset_d];
    int32_t kernel_d   = conv_dimension == 3 ? weights[offset_kernel] : 1;
    int32_t kernel_h   = weights[offset_kernel + offset_d];
    int32_t kernel_w   = weights[offset_kernel + 1 + offset_d];
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

    if ((full_H % stride_y != 0) || (full_W % stride_x != 0) || (full_D % stride_d != 0))
    {
        msg = "Parameters must yield exact integer output dimensions";
        return 1;
    }

    if ((OH != (full_H / stride_y) + 1) || (OW != (full_W / stride_x) + 1) || (OD != (full_D / stride_d) + 1))
    {
        std::string msg_d = "";
        if (conv_dimension == 3)
        {
            msg_d += std::to_string((full_D / stride_d) + 1) + ",";
        }
        msg = "Mismatch between output shape provided and expected output shape (" + msg_d +
              std::to_string((full_H / stride_y) + 1) + "," + std::to_string((full_W / stride_x) + 1) + ")";
        return 1;
    }

    if (InDtype != TOSA_REF_TYPE_INT8 && attribute->input_zp() != 0)
    {
        msg = "Input zero point must be zero for non-int8 data";
        return 1;
    }
    if (WeightDtype != TOSA_REF_TYPE_INT8 && attribute->weight_zp() != 0)
    {
        msg = "Weight zero point must be zero for non-int8 data";
        return 1;
    }

    return 0;
}

int check_fft_shape(const std::vector<int32_t>& in_real,
                    const std::vector<int32_t>& in_imag,
                    const std::vector<int32_t>& out_real,
                    const std::vector<int32_t>& out_imag,
                    std::string& msg)
{
    const bool is_rfft   = in_imag.empty();
    auto is_power_of_two = [](int32_t n) -> bool { return (n & (n - 1)) == 0 && n > 0; };

    if (!is_power_of_two(in_real[1]) || !is_power_of_two(in_real[2]))
    {
        msg = "Input height and width must be a power of two";
        return 1;
    }

    // RFFT does not have a second input
    if (!is_rfft)
    {
        bool input_check = true;
        for (size_t i = 0; i < in_real.size(); i++)
        {
            if (in_real[i] != in_imag[i])
            {
                input_check = false;
                break;
            }
        }
        if (!input_check)
        {
            msg = "Mismatch between real input shape and imaginary input shape";
            return 1;
        }
    }

    bool output_check = true;
    for (size_t i = 0; i < out_real.size(); i++)
    {
        if (out_real[i] != out_imag[i])
        {
            output_check = false;
            break;
        }
    }
    if (!output_check)
    {
        msg = "Mismatch between real output shape and imaginary output shape";
        return 1;
    }

    if (in_real[0] != out_real[0])
    {
        msg = "Input and output batch size don't match";
        return 1;
    }
    if (in_real[1] != out_real[1])
    {
        msg = "Input and output height don't match";
        return 1;
    }

    if (is_rfft)
    {
        if (in_real[2] / 2 + 1 != out_real[2])
        {
            msg = "Output width is expected to match input width / 2 + 1";
            return 1;
        }
    }
    else
    {
        if (in_real[2] != out_real[2])
        {
            msg = "Input and output width don't match";
            return 1;
        }
    }

    return 0;
}

template <int Rank, TOSA_REF_TYPE Dtype>
OpArgMax<Rank, Dtype>::OpArgMax(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_)
    : GraphNode(sgt_, Op_ARGMAX, id_)
{
    setRequiredOperands(1, 1);
    setRequiredRank(1);

    INIT_ATTRIBUTE(Axis);
}

template <int Rank, TOSA_REF_TYPE Dtype>
OpArgMax<Rank, Dtype>::~OpArgMax()
{
    if (attribute)
        delete attribute;
}

template <int Rank, TOSA_REF_TYPE Dtype>
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

    if (outputs[0]->getDtype() != TOSA_REF_TYPE_INT32)
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

    if (validateNanMode(attribute->nan_mode()))
    {
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

template <int Rank, TOSA_REF_TYPE Dtype>
int OpArgMax<Rank, Dtype>::eval()
{
    // Check Tosa Level
    auto tosa_level = g_func_config.tosa_level;
    LEVEL_CHECK(Rank <= tosa_level.MAX_RANK, "Rank should be smaller than or equal to MAX_RANK");

    Eigen::Tensor<InEigenType, Rank> input = this->input->getTensor();

    Eigen::array<DenseIndex, Rank> shuffle_indices;
    shuffle_indices[0] = attribute->axis();

    for (int i = 1; i < Rank; i++)
    {
        shuffle_indices[i] = (i <= attribute->axis()) ? i - 1 : i;
    }

    auto dimensions = input.dimensions();

    Eigen::array<long, 2> matrix_dimensions;
    matrix_dimensions[0] = dimensions[attribute->axis()];
    matrix_dimensions[1] = 1;
    for (int i = 0; i < Rank; i++)
    {
        if (i == attribute->axis())
            continue;
        matrix_dimensions[1] *= dimensions[i];
    }

    // Put the reduction axis as the first dimension and reshape to a matrix where each value in the second
    // dimension represents a position in the output tensor
    Eigen::Tensor<InEigenType, 2> shuffled_input = input.shuffle(shuffle_indices).reshape(matrix_dimensions);

    Eigen::Tensor<OutEigenType, 1> argmaxes(matrix_dimensions[1]);

    constexpr bool is_fp = std::is_floating_point_v<InEigenType>;
    const auto nan_mode  = attribute->nan_mode();

    // Find the maximum of a row in the matrix.
    for (DenseIndex j = 0; j < matrix_dimensions[1]; j++)
    {
        InEigenType max_val =
            (is_fp && isIgnoringNan(nan_mode)) ? DtypeLimits<Dtype>::quiet_NaN : DtypeLimits<Dtype>::low_extreme;

        OutEigenType max_idx = 0;

        for (OutEigenType i = 0; i < matrix_dimensions[0]; i++)
        {
            InEigenType val    = shuffled_input(i, j);
            InEigenType result = applyMax<InEigenType>(val, max_val, nan_mode);
            if (result != max_val)
            {
                // If there are NaNs, return the first NaN position.
                if (!(is_fp && std::isnan(result) && std::isnan(max_val)))
                {
                    max_val = result;
                    max_idx = i;
                }
            }
        }

        argmaxes(j) = max_idx;
    }

    Eigen::array<long, Rank - 1> output_shape;
    DenseIndex in_idx  = 0;
    DenseIndex out_idx = 0;
    while (in_idx < Rank)
    {
        if (in_idx != attribute->axis())
        {
            output_shape[out_idx] = dimensions[in_idx];
            out_idx++;
        }
        in_idx++;
    }

    // Reshape to the original dimensions without the reduction axis.
    this->output->getTensor() = argmaxes.reshape(output_shape);

    return GraphNode::eval();
}

template <TOSA_REF_TYPE Dtype, TOSA_REF_TYPE AccDtype>
OpAvgPool2d<Dtype, AccDtype>::OpAvgPool2d(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_)
    : GraphNode(sgt_, Op_AVG_POOL2D, id_)
{
    setRequiredOperands(1, 1);
    setRequiredRank(4, 4);

    INIT_ATTRIBUTE(Pool);
}

template <TOSA_REF_TYPE Dtype, TOSA_REF_TYPE AccDtype>
OpAvgPool2d<Dtype, AccDtype>::~OpAvgPool2d()
{
    if (attribute)
        delete attribute;
}

template <TOSA_REF_TYPE Dtype, TOSA_REF_TYPE AccDtype>
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

    ERROR_IF(Dtype != TOSA_REF_TYPE_INT8 && attribute->input_zp() != 0,
             "OpAvgPool2d: Input zeropoint must be zero for non int8_t data");
    ERROR_IF(Dtype != TOSA_REF_TYPE_INT8 && attribute->output_zp() != 0,
             "OpAvgPool2d: Output zeropoint must be zero for non int8_t data");

    std::string msg;
    if (check_pool2d_attribute(attribute, in->getShape(), out->getShape(), msg))
    {
        msg = "OpAvgPool2d: " + msg;
        printNodeValidationError(msg.c_str());
        return 1;
    }

    return 0;
}

// assuming input and output tensor have same scales like tflite reference
// so no need to scale input and output
template <TOSA_REF_TYPE Dtype, TOSA_REF_TYPE AccDtype>
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
    int kernel_y   = this->attribute->kernel()[0];
    int kernel_x   = this->attribute->kernel()[1];
    int stride_y   = this->attribute->stride()[0];
    int stride_x   = this->attribute->stride()[1];

    // Check Tosa Level
    auto tosa_level = g_func_config.tosa_level;
    LEVEL_CHECK(kernel_y <= tosa_level.MAX_KERNEL, "kernel_y should be smaller than or equal to MAX_KERNEL");
    LEVEL_CHECK(kernel_x <= tosa_level.MAX_KERNEL, "kernel_x should be smaller than or equal to MAX_KERNEL");
    LEVEL_CHECK(stride_y <= tosa_level.MAX_STRIDE, "stride_y should be smaller than or equal to MAX_STRIDE");
    LEVEL_CHECK(stride_x <= tosa_level.MAX_STRIDE, "stride_x should be smaller than or equal to MAX_STRIDE");
    LEVEL_CHECK(pad_top <= tosa_level.MAX_KERNEL, "pad_top should be smaller than or equal to MAX_KERNEL");
    LEVEL_CHECK(pad_bottom <= tosa_level.MAX_KERNEL, "pad_bottom should be smaller than or equal to MAX_KERNEL");
    LEVEL_CHECK(pad_left <= tosa_level.MAX_KERNEL, "pad_left should be smaller than or equal to MAX_KERNEL");
    LEVEL_CHECK(pad_right <= tosa_level.MAX_KERNEL, "pad_right should be smaller than or equal to MAX_KERNEL");

    TOSA_REF_TYPE accum_dtype = ConvertDType(this->attribute->acc_type());

    DEBUG_INFO(OP,
               "perform AvgPool2d, input.shape=[%d,%d,%d,%d], output.shape=[%d,%d,%d,%d], kernel=[%d,%d], "
               "stride=[%d,%d], pad=[%d,%d,%d,%d], acc_type=%s",
               in_batch, in_height, in_width, in_channels, out_batch, out_height, out_width, out_channels, kernel_y,
               kernel_x, stride_y, stride_x, pad_top, pad_bottom, pad_left, pad_right, EnumNamesDType()[accum_dtype]);

    Eigen::array<std::pair<int32_t, int32_t>, 4> pad;
    pad[0] = std::make_pair(0, 0);
    pad[1] = std::make_pair(pad_top, pad_bottom);
    pad[2] = std::make_pair(pad_left, pad_right);
    pad[3] = std::make_pair(0, 0);

    ETensor4<InEigenType> input_val = this->in->getTensor();
    if (Dtype == TOSA_REF_TYPE_INT8)
    {
        input_val = input_val - (InEigenType)attribute->input_zp();
    }

    if (g_func_config.abs_mode)
    {
        // in abs_mode: take abs values of input_val
        input_val = input_val.abs();
    }

    // assuming input and output have same scales
    // so input and output scaling is not required
    // TODO: check if this assumption TOSA made

    ETensor4<OutEigenType> out_tens(out_batch, out_height, out_width, out_channels);

    // sum pool
    for (int ob = 0; ob < out_batch; ++ob)
    {
        for (int oh = 0; oh < out_height; ++oh)
        {
            for (int ow = 0; ow < out_width; ++ow)
            {
                for (int oc = 0; oc < out_channels; ++oc)
                {
                    AccEigenType acc(0);
                    int filter_count = 0;
                    const int iy     = oh * stride_y - pad_top;
                    const int ix     = ow * stride_x - pad_left;
                    for (int ky = 0; ky < kernel_y; ++ky)
                    {
                        for (int kx = 0; kx < kernel_x; ++kx)
                        {
                            const int y = iy + ky;
                            const int x = ix + kx;
                            if ((0 <= y && y < in_height) && (0 <= x && x < in_width))
                            {
                                ++filter_count;
                                acc = acc + (AccEigenType)input_val(ob, y, x, oc);
                            }
                        }
                    }
                    if (Dtype != TOSA_REF_TYPE_FP32 && Dtype != TOSA_REF_TYPE_FP16 && Dtype != TOSA_REF_TYPE_BF16 &&
                        Dtype != TOSA_REF_TYPE_FP64 && Dtype != TOSA_REF_TYPE_FP8E4M3 && Dtype != TOSA_REF_TYPE_FP8E5M2)
                    {
                        try
                        {
                            int32_t multiplier, shift;
                            OutEigenType out;
                            TosaReference::QuantUtil::reciprocal_scale(filter_count, multiplier, shift);

                            out = (OutEigenType)TosaReference::QuantUtil::apply_scale_32(acc, multiplier, shift, false);
                            out = out + (OutEigenType)(attribute->output_zp());
                            out = std::max(out, (OutEigenType)QMin);
                            out_tens(ob, oh, ow, oc) = std::min(out, (OutEigenType)QMax);
                        }
                        catch (std::string desc)
                        {
                            REQUIRE(false, "OpAvgPool2d apply_scale_32() fails: %s.", desc.c_str());
                        }
                    }
                    else
                    {
                        REQUIRE(filter_count != 0, "OpAvgPool2d number of filters should be non-zero.");
                        out_tens(ob, oh, ow, oc) =
                            static_cast<OutEigenType>(acc / static_cast<AccEigenType>(filter_count));
                    }
                }
            }
        }
    }
    this->out->getTensor() = out_tens;
    return GraphNode::eval();
}

template <TOSA_REF_TYPE InDtype, TOSA_REF_TYPE WeightDtype, TOSA_REF_TYPE AccDtype, TOSA_REF_TYPE OutDtype>
OpConv2d<InDtype, WeightDtype, AccDtype, OutDtype>::OpConv2d(SubgraphTraverser* sgt_,
                                                             TosaAttributeBase* attribute_,
                                                             uint64_t id_)
    : GraphNode(sgt_, Op_CONV2D, id_)
{
    setRequiredOperands(3, 1);
    setRequiredRank(4, 4);

    INIT_ATTRIBUTE(Conv);
}

template <TOSA_REF_TYPE InDtype, TOSA_REF_TYPE WeightDtype, TOSA_REF_TYPE AccDtype, TOSA_REF_TYPE OutDtype>
OpConv2d<InDtype, WeightDtype, AccDtype, OutDtype>::~OpConv2d()
{
    if (attribute)
        delete attribute;
}

template <TOSA_REF_TYPE InDtype, TOSA_REF_TYPE WeightDtype, TOSA_REF_TYPE AccDtype, TOSA_REF_TYPE OutDtype>
int OpConv2d<InDtype, WeightDtype, AccDtype, OutDtype>::checkTensorAttributes()
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

template <TOSA_REF_TYPE InDtype, TOSA_REF_TYPE WeightDtype, TOSA_REF_TYPE AccDtype, TOSA_REF_TYPE OutDtype>
int OpConv2d<InDtype, WeightDtype, AccDtype, OutDtype>::eval()
{
    int in_batch    = this->input->getShape()[0];
    int in_height   = this->input->getShape()[1];
    int in_width    = this->input->getShape()[2];
    int in_channels = this->input->getShape()[3];

    int k_out_channels = this->weight->getShape()[0];
    int k_height       = this->weight->getShape()[1];
    int k_width        = this->weight->getShape()[2];
    int k_in_channels  = this->weight->getShape()[3];

    int b_out_channels = this->bias->getShape()[0];

    int out_batch    = this->output->getShape()[0];
    int out_height   = this->output->getShape()[1];
    int out_width    = this->output->getShape()[2];
    int out_channels = this->output->getShape()[3];

    ERROR_IF(in_batch != out_batch, "OpConv2d: tensor batch mismatch %d != %d", in_batch, out_batch);
    ERROR_IF(k_in_channels != in_channels, "OpConv2d: tensor input channel mismatch %d != %d", k_in_channels,
             in_channels);
    ERROR_IF(k_out_channels != out_channels, "OpConv2d: tensor output channel mismatch %d != %d", k_out_channels,
             out_channels);
    ERROR_IF(b_out_channels != out_channels && b_out_channels != 1, "OpConv2d: bias channel mismatch %d != %d",
             b_out_channels, out_channels);

    int pad_top    = this->attribute->pad()[0];
    int pad_bottom = this->attribute->pad()[1];
    int pad_left   = this->attribute->pad()[2];
    int pad_right  = this->attribute->pad()[3];

    int stride_y   = this->attribute->stride()[0];
    int stride_x   = this->attribute->stride()[1];
    int dilation_y = this->attribute->dilation()[0];
    int dilation_x = this->attribute->dilation()[1];

    // Check Tosa Level
    auto tosa_level = g_func_config.tosa_level;
    LEVEL_CHECK(dilation_y * k_height <= tosa_level.MAX_KERNEL,
                "dilation_y * KH should be smaller than or equal to MAX_KERNEL");
    LEVEL_CHECK(dilation_x * k_width <= tosa_level.MAX_KERNEL,
                "dilation_x * KW should be smaller than or equal to MAX_KERNEL");
    LEVEL_CHECK(pad_top <= tosa_level.MAX_KERNEL, "pad_top should be smaller than or equal to MAX_KERNEL");
    LEVEL_CHECK(pad_bottom <= tosa_level.MAX_KERNEL, "pad_bottom should be smaller than or equal to MAX_KERNEL");
    LEVEL_CHECK(pad_left <= tosa_level.MAX_KERNEL, "pad_left should be smaller than or equal to MAX_KERNEL");
    LEVEL_CHECK(pad_right <= tosa_level.MAX_KERNEL, "pad_right should be smaller than or equal to MAX_KERNEL");
    LEVEL_CHECK(stride_y <= tosa_level.MAX_STRIDE, "stride_y should be smaller than or equal to MAX_STRIDE");
    LEVEL_CHECK(stride_x <= tosa_level.MAX_STRIDE, "stride_x should be smaller than or equal to MAX_STRIDE");

    DEBUG_INFO(OP,
               "perform OpConv2d, input.shape=[%d,%d,%d,%d], weight.shape=[%d,%d,%d,%d], output.shape=[%d,%d,%d,%d], "
               "stride=[%d,%d], dilation=[%d,%d], pad=[%d,%d,%d,%d]",
               in_batch, in_height, in_width, in_channels, k_height, k_width, k_in_channels, k_out_channels, out_batch,
               out_height, out_width, out_channels, stride_y, stride_x, dilation_y, dilation_x, pad_top, pad_bottom,
               pad_left, pad_right);

    Eigen::array<std::pair<int32_t, int32_t>, 4> pad;
    pad[0] = std::make_pair(0, 0);
    pad[1] = std::make_pair(pad_top, pad_bottom);
    pad[2] = std::make_pair(pad_left, pad_right);
    pad[3] = std::make_pair(0, 0);

    TIn input_val      = this->input->getTensor();
    TWeight weight_val = this->weight->getTensor();
    if (InDtype == TOSA_REF_TYPE_INT8 || WeightDtype == TOSA_REF_TYPE_INT8)
    {
        input_val  = input_val - (InEigenType)attribute->input_zp();
        weight_val = weight_val - (WeightEigenType)attribute->weight_zp();
    }

    TBias bias_val = this->bias->getTensor();

    if (g_func_config.abs_mode)
    {
        // in abs_mode: take abs values of conv operands
        input_val  = input_val.abs();
        weight_val = weight_val.abs();
        bias_val   = bias_val.abs();

        if (!this->attribute->local_bound())
        {
            Eigen::Tensor<InEigenType, 0> input_abs_max = input_val.maximum();
            input_val.setConstant(input_abs_max(0));
        }
    }

    ETensor4<InEigenType> input_padded = input_val.pad(pad);

    // 1. initialize with bias
    Eigen::array<Eigen::Index, 4> reshape_dim;
    reshape_dim.fill(1);
    reshape_dim[3] = b_out_channels;

    Eigen::array<Eigen::Index, 4> bcast;
    bcast[0]                  = out_batch;
    bcast[1]                  = out_height;
    bcast[2]                  = out_width;
    bcast[3]                  = (b_out_channels == 1) ? out_channels : 1;
    this->output->getTensor() = bias_val.reshape(reshape_dim).broadcast(bcast);

    // 2. direct convolution
    int iy_pad, ix_pad;

    for (int ob = 0; ob < out_batch; ob++)
    {
        for (int oy = 0; oy < out_height; oy++)
        {
            for (int ox = 0; ox < out_width; ox++)
            {
                for (int oc = 0; oc < out_channels; oc++)
                {
                    AccEigenType acc(0.0);
                    for (int ky = 0; ky < k_height; ky++)
                    {
                        iy_pad = oy * stride_y + ky * dilation_y;
                        for (int kx = 0; kx < k_width; kx++)
                        {
                            ix_pad = ox * stride_x + kx * dilation_x;

                            // derive x, y indices into original input tensor
                            int y         = iy_pad - pad_top;
                            int x         = ix_pad - pad_left;
                            bool in_scope = (0 <= x && x < in_width) && (0 <= y && y < in_height);
                            if (!in_scope && !g_func_config.tosaExtraMultiplies())
                            {
                                // no need to do multiply and accumulate
                                continue;
                            }
                            for (int ic = 0; ic < in_channels; ic++)
                            {
                                acc += (static_cast<AccEigenType>(input_padded(ob, iy_pad, ix_pad, ic)) *
                                        static_cast<AccEigenType>(weight_val(oc, ky, kx, ic)));
                            }
                        }
                    }

                    // add bias to accumulated value
                    OutEigenType bias                         = this->output->getTensor()(ob, oy, ox, oc);
                    this->output->getTensor()(ob, oy, ox, oc) = bias + static_cast<OutEigenType>(acc);
                }
            }
        }
    }

    if (OutDtype == TOSA_REF_TYPE_INT48)
    {
        this->output->getTensor() = this->output->getTensor().cwiseMax((OutEigenType)AccQMin);
        this->output->getTensor() = this->output->getTensor().cwiseMin((OutEigenType)AccQMax);
    }

    return GraphNode::eval();
}

template <TOSA_REF_TYPE InDtype, TOSA_REF_TYPE WeightDtype, TOSA_REF_TYPE AccDtype, TOSA_REF_TYPE OutDtype>
OpConv3d<InDtype, WeightDtype, AccDtype, OutDtype>::OpConv3d(SubgraphTraverser* sgt_,
                                                             TosaAttributeBase* attribute_,
                                                             uint64_t id_)
    : GraphNode(sgt_, Op_CONV3D, id_)
{
    setRequiredOperands(3, 1);
    setRequiredRank(5, 5);

    INIT_ATTRIBUTE(Conv);
}

template <TOSA_REF_TYPE InDtype, TOSA_REF_TYPE WeightDtype, TOSA_REF_TYPE AccDtype, TOSA_REF_TYPE OutDtype>
OpConv3d<InDtype, WeightDtype, AccDtype, OutDtype>::~OpConv3d()
{
    if (attribute)
        delete attribute;
}

template <TOSA_REF_TYPE InDtype, TOSA_REF_TYPE WeightDtype, TOSA_REF_TYPE AccDtype, TOSA_REF_TYPE OutDtype>
int OpConv3d<InDtype, WeightDtype, AccDtype, OutDtype>::checkTensorAttributes()
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

template <TOSA_REF_TYPE InDtype, TOSA_REF_TYPE WeightDtype, TOSA_REF_TYPE AccDtype, TOSA_REF_TYPE OutDtype>
int OpConv3d<InDtype, WeightDtype, AccDtype, OutDtype>::eval()
{
    int in_batch    = this->input->getShape()[0];
    int in_depth    = this->input->getShape()[1];
    int in_height   = this->input->getShape()[2];
    int in_width    = this->input->getShape()[3];
    int in_channels = this->input->getShape()[4];

    int k_out_channels = this->weight->getShape()[0];
    int k_depth        = this->weight->getShape()[1];
    int k_height       = this->weight->getShape()[2];
    int k_width        = this->weight->getShape()[3];
    int k_in_channels  = this->weight->getShape()[4];

    int b_out_channels = this->bias->getShape()[0];

    int out_batch    = this->output->getShape()[0];
    int out_depth    = this->output->getShape()[1];
    int out_height   = this->output->getShape()[2];
    int out_width    = this->output->getShape()[3];
    int out_channels = this->output->getShape()[4];

    ERROR_IF(in_batch != out_batch, "OpConv3d: tensor batch mismatch %d != %d", in_batch, out_batch);
    ERROR_IF(k_in_channels != in_channels, "OpConv3d: tensor input channel mismatch %d != %d", k_in_channels,
             in_channels);
    ERROR_IF(k_out_channels != out_channels, "OpConv3d: tensor output channel mismatch %d != %d", k_out_channels,
             out_channels);
    ERROR_IF(b_out_channels != out_channels && b_out_channels != 1, "OpConv3d: bias channel mismatch %d != %d",
             b_out_channels, out_channels);

    int pad_d0     = this->attribute->pad()[0];
    int pad_d1     = this->attribute->pad()[1];
    int pad_top    = this->attribute->pad()[2];
    int pad_bottom = this->attribute->pad()[3];
    int pad_left   = this->attribute->pad()[4];
    int pad_right  = this->attribute->pad()[5];

    int stride_d = this->attribute->stride()[0];
    int stride_y = this->attribute->stride()[1];
    int stride_x = this->attribute->stride()[2];

    int dilation_d = this->attribute->dilation()[0];
    int dilation_y = this->attribute->dilation()[1];
    int dilation_x = this->attribute->dilation()[2];

    // Check Tosa Level
    auto tosa_level = g_func_config.tosa_level;
    LEVEL_CHECK(dilation_d * k_depth <= tosa_level.MAX_KERNEL,
                "dilation_d * KD should be smaller than or equal to MAX_KERNEL");
    LEVEL_CHECK(dilation_y * k_height <= tosa_level.MAX_KERNEL,
                "dilation_y * KH should be smaller than or equal to MAX_KERNEL");
    LEVEL_CHECK(dilation_x * k_width <= tosa_level.MAX_KERNEL,
                "dilation_x * KW should be smaller than or equal to MAX_KERNEL");
    LEVEL_CHECK(pad_d0 <= tosa_level.MAX_KERNEL, "pad_d0 should be smaller than or equal to MAX_KERNEL");
    LEVEL_CHECK(pad_d1 <= tosa_level.MAX_KERNEL, "pad_d1 should be smaller than or equal to MAX_KERNEL");
    LEVEL_CHECK(pad_top <= tosa_level.MAX_KERNEL, "pad_top should be smaller than or equal to MAX_KERNEL");
    LEVEL_CHECK(pad_bottom <= tosa_level.MAX_KERNEL, "pad_bottom should be smaller than or equal to MAX_KERNEL");
    LEVEL_CHECK(pad_left <= tosa_level.MAX_KERNEL, "pad_left should be smaller than or equal to MAX_KERNEL");
    LEVEL_CHECK(pad_right <= tosa_level.MAX_KERNEL, "pad_right should be smaller than or equal to MAX_KERNEL");
    LEVEL_CHECK(stride_y <= tosa_level.MAX_STRIDE, "stride_y should be smaller than or equal to MAX_STRIDE");
    LEVEL_CHECK(stride_x <= tosa_level.MAX_STRIDE, "stride_x should be smaller than or equal to MAX_STRIDE");
    LEVEL_CHECK(stride_d <= tosa_level.MAX_STRIDE, "stride_d should be smaller than or equal to MAX_STRIDE");

    DEBUG_INFO(
        OP,
        "perform OpConv3d, input.shape=[%d,%d,%d,%d,%d], weight.shape=[%d,%d,%d,%d,%d], output.shape=[%d,%d,%d,%d,%d], "
        "stride=[%d,%d,%d], dilation=[%d,%d,%d], pad=[%d,%d,%d,%d,%d,%d]",
        in_batch, in_depth, in_height, in_width, in_channels, k_out_channels, k_depth, k_height, k_width, k_in_channels,
        out_batch, out_depth, out_height, out_width, out_channels, stride_d, stride_y, stride_x, dilation_d, dilation_y,
        dilation_x, pad_d0, pad_d1, pad_top, pad_bottom, pad_left, pad_right);

    Eigen::array<std::pair<int32_t, int32_t>, 5> pad;
    pad[0] = std::make_pair(0, 0);
    pad[1] = std::make_pair(pad_d0, pad_d1);
    pad[2] = std::make_pair(pad_top, pad_bottom);
    pad[3] = std::make_pair(pad_left, pad_right);
    pad[4] = std::make_pair(0, 0);

    TIn input_val      = this->input->getTensor();
    TWeight weight_val = this->weight->getTensor();
    if (InDtype == TOSA_REF_TYPE_INT8 || WeightDtype == TOSA_REF_TYPE_INT8)
    {
        input_val  = input_val - (InEigenType)attribute->input_zp();
        weight_val = weight_val - (WeightEigenType)attribute->weight_zp();
    }

    TBias bias_val = this->bias->getTensor();

    if (g_func_config.abs_mode)
    {
        // in abs_mode: take abs values of conv operands
        input_val  = input_val.abs();
        weight_val = weight_val.abs();
        bias_val   = bias_val.abs();

        if (!this->attribute->local_bound())
        {
            Eigen::Tensor<InEigenType, 0> input_abs_max = input_val.maximum();
            input_val.setConstant(input_abs_max(0));
        }
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
    bcast[4]                  = (b_out_channels == 1) ? out_channels : 1;
    this->output->getTensor() = bias_val.reshape(reshape_dim).broadcast(bcast);

    // 2. direct convolution
    int id_pad, iy_pad, ix_pad;

    for (int ob = 0; ob < out_batch; ob++)
    {
        for (int od = 0; od < out_depth; od++)
        {
            for (int oy = 0; oy < out_height; oy++)
            {
                for (int ox = 0; ox < out_width; ox++)
                {
                    for (int oc = 0; oc < out_channels; oc++)
                    {
                        AccEigenType acc(0.0);
                        for (int kd = 0; kd < k_depth; kd++)
                        {
                            id_pad = od * stride_d + kd * dilation_d;
                            for (int ky = 0; ky < k_height; ky++)
                            {
                                iy_pad = oy * stride_y + ky * dilation_y;
                                for (int kx = 0; kx < k_width; kx++)
                                {
                                    ix_pad = ox * stride_x + kx * dilation_x;

                                    // derive x, y, d indices into original input tensor
                                    int d         = id_pad - pad_d0;
                                    int y         = iy_pad - pad_top;
                                    int x         = ix_pad - pad_left;
                                    bool in_scope = (0 <= x && x < in_width) && (0 <= y && y < in_height) &&
                                                    (0 <= d && d < in_depth);
                                    if (!in_scope && !g_func_config.tosaExtraMultiplies())
                                    {
                                        // no need to do multiply and accumulate
                                        continue;
                                    }

                                    for (int ic = 0; ic < in_channels; ic++)
                                    {
                                        acc += ((AccEigenType)input_padded(ob, id_pad, iy_pad, ix_pad, ic) *
                                                (AccEigenType)weight_val(oc, kd, ky, kx, ic));
                                    }
                                }
                            }
                        }
                        // add bias to accumulated value
                        OutEigenType bias                             = this->output->getTensor()(ob, od, oy, ox, oc);
                        this->output->getTensor()(ob, od, oy, ox, oc) = bias + (OutEigenType)acc;
                    }
                }
            }
        }
    }

    if (OutDtype == TOSA_REF_TYPE_INT48)
    {
        this->output->getTensor() = this->output->getTensor().cwiseMax((OutEigenType)AccQMin);
        this->output->getTensor() = this->output->getTensor().cwiseMin((OutEigenType)AccQMax);
    }

    return GraphNode::eval();
}

template <TOSA_REF_TYPE InDtype, TOSA_REF_TYPE WeightDtype, TOSA_REF_TYPE AccDtype, TOSA_REF_TYPE OutDtype>
OpDepthwiseConv2d<InDtype, WeightDtype, AccDtype, OutDtype>::OpDepthwiseConv2d(SubgraphTraverser* sgt_,
                                                                               TosaAttributeBase* attribute_,
                                                                               uint64_t id_)
    : GraphNode(sgt_, Op_DEPTHWISE_CONV2D, id_)
{
    setRequiredOperands(3, 1);
    setRequiredRank(4, 4);

    INIT_ATTRIBUTE(Conv);
}

template <TOSA_REF_TYPE InDtype, TOSA_REF_TYPE WeightDtype, TOSA_REF_TYPE AccDtype, TOSA_REF_TYPE OutDtype>
OpDepthwiseConv2d<InDtype, WeightDtype, AccDtype, OutDtype>::~OpDepthwiseConv2d()
{
    if (attribute)
        delete attribute;
}

template <TOSA_REF_TYPE InDtype, TOSA_REF_TYPE WeightDtype, TOSA_REF_TYPE AccDtype, TOSA_REF_TYPE OutDtype>
int OpDepthwiseConv2d<InDtype, WeightDtype, AccDtype, OutDtype>::checkTensorAttributes()
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

template <TOSA_REF_TYPE InDtype, TOSA_REF_TYPE WeightDtype, TOSA_REF_TYPE AccDtype, TOSA_REF_TYPE OutDtype>
int OpDepthwiseConv2d<InDtype, WeightDtype, AccDtype, OutDtype>::eval()
{
    int in_batch    = this->input->getShape()[0];
    int in_height   = this->input->getShape()[1];
    int in_width    = this->input->getShape()[2];
    int in_channels = this->input->getShape()[3];

    int k_height      = this->weight->getShape()[0];
    int k_width       = this->weight->getShape()[1];
    int k_in_channels = this->weight->getShape()[2];
    int k_multiplier  = this->weight->getShape()[3];

    int b_out_channels = this->bias->getShape()[0];

    int out_batch    = this->output->getShape()[0];
    int out_height   = this->output->getShape()[1];
    int out_width    = this->output->getShape()[2];
    int out_channels = this->output->getShape()[3];

    ERROR_IF(in_batch != out_batch, "OpDepthwiseConv2d: tensor batch mismatch %d != %d", in_batch, out_batch);
    ERROR_IF(k_in_channels != in_channels, "OpDepthwiseConv2d: tensor input channel mismatch %d != %d", k_in_channels,
             in_channels);
    ERROR_IF(in_channels * k_multiplier != out_channels, "OpDepthwiseConv2d: tensor output channel mismatch %d != %d",
             in_channels * k_multiplier, out_channels);
    ERROR_IF(b_out_channels != out_channels && b_out_channels != 1,
             "OpDepthwiseConv2d: bias channels mismatch %d != %d", b_out_channels, out_channels);

    int pad_top    = this->attribute->pad()[0];
    int pad_bottom = this->attribute->pad()[1];
    int pad_left   = this->attribute->pad()[2];
    int pad_right  = this->attribute->pad()[3];

    int stride_y   = this->attribute->stride()[0];
    int stride_x   = this->attribute->stride()[1];
    int dilation_y = this->attribute->dilation()[0];
    int dilation_x = this->attribute->dilation()[1];

    // Check Tosa Level
    auto tosa_level = g_func_config.tosa_level;
    LEVEL_CHECK(dilation_y * k_height <= tosa_level.MAX_KERNEL,
                "dilation_y * KH should be smaller than or equal to MAX_KERNEL");
    LEVEL_CHECK(dilation_x * k_width <= tosa_level.MAX_KERNEL,
                "dilation_x * KW should be smaller than or equal to MAX_KERNEL");
    LEVEL_CHECK(pad_top <= tosa_level.MAX_KERNEL, "pad_top should be smaller than or equal to MAX_KERNEL");
    LEVEL_CHECK(pad_bottom <= tosa_level.MAX_KERNEL, "pad_bottom should be smaller than or equal to MAX_KERNEL");
    LEVEL_CHECK(pad_left <= tosa_level.MAX_KERNEL, "pad_left should be smaller than or equal to MAX_KERNEL");
    LEVEL_CHECK(pad_right <= tosa_level.MAX_KERNEL, "pad_right should be smaller than or equal to MAX_KERNEL");
    LEVEL_CHECK(stride_y <= tosa_level.MAX_STRIDE, "stride_y should be smaller than or equal to MAX_STRIDE");
    LEVEL_CHECK(stride_x <= tosa_level.MAX_STRIDE, "stride_x should be smaller than or equal to MAX_STRIDE");

    DEBUG_INFO(OP,
               "perform OpDepthwiseConv2d, input.shape=[%d,%d,%d,%d], weight.shape=[%d,%d,%d,%d], "
               "output.shape=[%d,%d,%d,%d], stride=[%d,%d], dilation=[%d,%d], pad=[%d,%d,%d,%d]",
               in_batch, in_height, in_width, in_channels, k_height, k_width, k_in_channels, k_multiplier, out_batch,
               out_height, out_width, out_channels, stride_y, stride_x, dilation_y, dilation_x, pad_top, pad_bottom,
               pad_left, pad_right);

    Eigen::array<std::pair<int32_t, int32_t>, 4> pad;
    pad[0] = std::make_pair(0, 0);
    pad[1] = std::make_pair(pad_top, pad_bottom);
    pad[2] = std::make_pair(pad_left, pad_right);
    pad[3] = std::make_pair(0, 0);

    TIn input_val      = this->input->getTensor();
    TWeight weight_val = this->weight->getTensor();
    if (InDtype == TOSA_REF_TYPE_INT8 || WeightDtype == TOSA_REF_TYPE_INT8)
    {
        input_val  = input_val - (InEigenType)attribute->input_zp();
        weight_val = weight_val - (WeightEigenType)attribute->weight_zp();
    }

    TBias bias_val = this->bias->getTensor();

    if (g_func_config.abs_mode)
    {
        // in abs_mode: take abs values of conv operands
        input_val  = input_val.abs();
        weight_val = weight_val.abs();
        bias_val   = bias_val.abs();

        if (!this->attribute->local_bound())
        {
            Eigen::Tensor<InEigenType, 0> input_abs_max = input_val.maximum();
            input_val.setConstant(input_abs_max(0));
        }
    }

    ETensor4<InEigenType> input_padded = input_val.pad(pad);

    // GEMM doesn't fit well with DepthwiseConv2d
    // 1. use extract_image_patches() to handle stride/dilation/pad
    // 2. perform direct convolution

    // 1. extract_image_patches() output [N, KH, KW, OH * OW, IC]
    ETensor5<InEigenType> input_extract_patches = input_padded.extract_image_patches(
        k_height, k_width, stride_y, stride_x, dilation_y, dilation_x, Eigen::PADDING_VALID);

    Eigen::array<Eigen::Index, 4> reshape_dim;
    reshape_dim.fill(1);
    reshape_dim[3] = b_out_channels;

    Eigen::array<Eigen::Index, 4> bcast;
    bcast[0] = out_batch;
    bcast[1] = out_height;
    bcast[2] = out_width;
    bcast[3] = (b_out_channels == 1) ? out_channels : 1;

    // initialize with bias
    this->output->getTensor() = bias_val.reshape(reshape_dim).broadcast(bcast);

    // 2. direct depthwise convolution
    for (int ob = 0; ob < out_batch; ob++)
    {
        for (int oy = 0; oy < out_height; oy++)
        {
            for (int ox = 0; ox < out_width; ox++)
            {
                for (int ic = 0; ic < in_channels; ic++)
                {
                    for (int km = 0; km < k_multiplier; km++)
                    {
                        AccEigenType acc(0.0);
                        int iy = oy * stride_y - pad_top;
                        int ix = ox * stride_x - pad_left;
                        for (int ky = 0; ky < k_height; ky++)
                        {
                            for (int kx = 0; kx < k_width; kx++)
                            {
                                int y         = iy + ky * dilation_y;
                                int x         = ix + kx * dilation_x;
                                bool in_scope = (0 <= y && y < in_height) && (0 <= x && x < in_width);
                                if (!in_scope && !g_func_config.tosaExtraMultiplies())
                                {
                                    // no need to do multiply and accumulate
                                    continue;
                                }
                                AccEigenType value(0.0);
                                if (in_scope)
                                {
                                    value = (AccEigenType)input_val(ob, y, x, ic);
                                }
                                // Perform multiplication in AccEigenType then cast to OutEigenType
                                acc += value * (AccEigenType)weight_val(ky, kx, ic, km);
                            }
                        }
                        this->output->getTensor()(ob, oy, ox, ic * k_multiplier + km) += (OutEigenType)acc;
                    }
                }
            }
        }
    }

    if (OutDtype == TOSA_REF_TYPE_INT48)
    {
        this->output->getTensor() = this->output->getTensor().cwiseMax((OutEigenType)AccQMin);
        this->output->getTensor() = this->output->getTensor().cwiseMin((OutEigenType)AccQMax);
    }

    return GraphNode::eval();
}

template <TOSA_REF_TYPE Input1Dtype, TOSA_REF_TYPE Input2Dtype, TOSA_REF_TYPE OutDtype>
OpMatMul<Input1Dtype, Input2Dtype, OutDtype>::OpMatMul(SubgraphTraverser* sgt_,
                                                       TosaAttributeBase* attribute_,
                                                       uint64_t id_)
    : GraphNode(sgt_, Op_MATMUL, id_)
{
    setRequiredOperands(2, 1);
    setRequiredRank(3, 3);

    INIT_ATTRIBUTE(MatMul);
}

template <TOSA_REF_TYPE Input1Dtype, TOSA_REF_TYPE Input2Dtype, TOSA_REF_TYPE OutDtype>
OpMatMul<Input1Dtype, Input2Dtype, OutDtype>::~OpMatMul()
{
    if (attribute)
        delete attribute;
}

template <TOSA_REF_TYPE Input1Dtype, TOSA_REF_TYPE Input2Dtype, TOSA_REF_TYPE OutDtype>
int OpMatMul<Input1Dtype, Input2Dtype, OutDtype>::checkTensorAttributes()
{
    if (validateRequiredOperands())
        return 1;

    if (validateRequiredRank(inputs[0]) || validateRequiredRank(inputs[1]) || validateRequiredRank(outputs[0]))
    {
        return 1;
    }

    ERROR_IF(outputs[0]->getDtype() != OutDtype,
             "OpMatMul: Output data type not supported for this configuration of operator");

    a      = dynamic_cast<TosaReference::TensorTemplate<TInput1>*>(inputs[0]);
    b      = dynamic_cast<TosaReference::TensorTemplate<TInput2>*>(inputs[1]);
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

    ERROR_IF(Input1Dtype != TOSA_REF_TYPE_INT8 && attribute->a_zp() != 0,
             "OpMatMul: A zeropoint must be zero for non int8_t data");
    ERROR_IF(Input2Dtype != TOSA_REF_TYPE_INT8 && attribute->b_zp() != 0,
             "OpMatMul: B zeropoint must be zero for non int8_t data");

    return 0;
}

template <TOSA_REF_TYPE Input1Dtype, TOSA_REF_TYPE Input2Dtype, TOSA_REF_TYPE OutDtype>
int OpMatMul<Input1Dtype, Input2Dtype, OutDtype>::eval()
{
    typedef Eigen::Tensor<int, 1>::DimensionPair DimPair;
    Eigen::array<DimPair, 1> dims{ { DimPair(1, 0) } };

    TInput1 a_val = this->a->getTensor();
    TInput2 b_val = this->b->getTensor();
    if (Input1Dtype == TOSA_REF_TYPE_INT8)
    {
        a_val = a_val - (Input1EigenType)attribute->a_zp();
    }
    if (Input2Dtype == TOSA_REF_TYPE_INT8)
    {
        b_val = b_val - (Input2EigenType)attribute->b_zp();
    }

    if (g_func_config.abs_mode)
    {
        // in abs_mode: take abs values of matmul operands
        a_val = a_val.abs();
        b_val = b_val.abs();
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

        TInput1Rank2 a_rank2_val = a_val.slice(a_begin_array, a_size_array).reshape(a_rank2_shape);
        TInput2Rank2 b_rank2_val = b_val.slice(b_begin_array, b_size_array).reshape(b_rank2_shape);
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

    if (OutDtype == TOSA_REF_TYPE_INT48)
    {
        this->output->getTensor() = this->output->getTensor().cwiseMax((OutEigenType)AccQMin);
        this->output->getTensor() = this->output->getTensor().cwiseMin((OutEigenType)AccQMax);
    }

    return GraphNode::eval();
}

template <TOSA_REF_TYPE Dtype>
OpMaxPool2d<Dtype>::OpMaxPool2d(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_)
    : GraphNode(sgt_, Op_MAX_POOL2D, id_)
{
    setRequiredOperands(1, 1);
    setRequiredRank(4, 4);

    INIT_ATTRIBUTE(Pool);
}

template <TOSA_REF_TYPE Dtype>
OpMaxPool2d<Dtype>::~OpMaxPool2d()
{
    if (attribute)
        delete attribute;
}

template <TOSA_REF_TYPE Dtype>
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

    if (GraphNode::validateNanMode(attribute->nan_mode()))
    {
        return 1;
    }

    return 0;
}

template <TOSA_REF_TYPE Dtype>
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

    int kernel_y = this->attribute->kernel()[0];
    int kernel_x = this->attribute->kernel()[1];
    int stride_y = this->attribute->stride()[0];
    int stride_x = this->attribute->stride()[1];

    // Check Tosa Level
    auto tosa_level = g_func_config.tosa_level;
    LEVEL_CHECK(kernel_y <= tosa_level.MAX_KERNEL, "kernel_y should be smaller than or equal to MAX_KERNEL");
    LEVEL_CHECK(kernel_x <= tosa_level.MAX_KERNEL, "kernel_x should be smaller than or equal to MAX_KERNEL");
    LEVEL_CHECK(stride_y <= tosa_level.MAX_STRIDE, "stride_y should be smaller than or equal to MAX_STRIDE");
    LEVEL_CHECK(stride_x <= tosa_level.MAX_STRIDE, "stride_x should be smaller than or equal to MAX_STRIDE");
    LEVEL_CHECK(pad_top <= tosa_level.MAX_KERNEL, "pad_top should be smaller than or equal to MAX_KERNEL");
    LEVEL_CHECK(pad_bottom <= tosa_level.MAX_KERNEL, "pad_bottom should be smaller than or equal to MAX_KERNEL");
    LEVEL_CHECK(pad_left <= tosa_level.MAX_KERNEL, "pad_left should be smaller than or equal to MAX_KERNEL");
    LEVEL_CHECK(pad_right <= tosa_level.MAX_KERNEL, "pad_right should be smaller than or equal to MAX_KERNEL");

    DEBUG_INFO(OP,
               "perform MaxPool2d, input.shape=[%d,%d,%d,%d], output.shape=[%d,%d,%d,%d], kernel=[%d,%d], "
               "stride=[%d,%d], pad=[%d,%d,%d,%d]",
               in_batch, in_height, in_width, in_channels, out_batch, out_height, out_width, out_channels, kernel_y,
               kernel_x, stride_y, stride_x, pad_top, pad_bottom, pad_left, pad_right);

    Eigen::array<Eigen::Index, 2> im2col_input_dims;
    im2col_input_dims[0] = kernel_y * kernel_x;
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

    // Set the padding value to be the lowest value that can be represented
    // by the datatype to ensure that any padding values will be equal
    // to or smaller than the actual maximum in the KH x KW patch.
    InEigenType padding_value = DtypeLimits<Dtype>::low_extreme;

    ETensor4<InEigenType> input_padded = this->in->getTensor().pad(pad, padding_value);

    // extract_image_patches() output [N, KH, KW, H * W, C]
    // transpose to [KH, KW, N, H * W, C]
    // reshape to [KH * KW, N * H * W * C]
    ETensor2<InEigenType> input_extract_patches =
        input_padded
            .extract_image_patches(kernel_y, kernel_x, stride_y, stride_x, 1, 1, Eigen::PADDING_VALID, padding_value)
            .shuffle(Eigen::array<Eigen::Index, 5>{ 1, 2, 0, 3, 4 })
            .reshape(im2col_input_dims);

    // 1D result with [N * H * W * C]
    ETensor1<OutEigenType> out_1d(this->out->getElementCount());

    // Get the maximum of the KHxHW patches along axis 0
    // Eigen's argmax behaves incorrectly on some cases with -inf and -max
    // so do it by hand.
    for (int j = 0; j < im2col_input_dims[1]; j++)
    {
        OutEigenType max = padding_value;
        for (int i = 0; i < im2col_input_dims[0]; i++)
        {
            OutEigenType val = input_extract_patches(i, j);
            max              = applyMax<OutEigenType>(max, val, attribute->nan_mode());
        }

        out_1d(j) = max;
    }

    // reshape result to [N, H, W, C]
    this->out->getTensor() = out_1d.reshape(col2im_output_dims);

    return GraphNode::eval();
}

template <TOSA_REF_TYPE Dtype>
OpFFT2d<Dtype>::OpFFT2d(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_)
    : GraphNode(sgt_, Op_FFT2D, id_)
{
    setRequiredOperands(2, 2);
    setRequiredRank(3, 3);

    INIT_ATTRIBUTE(FFT);
}

template <TOSA_REF_TYPE Dtype>
OpFFT2d<Dtype>::~OpFFT2d()
{
    if (attribute)
        delete attribute;
}

template <TOSA_REF_TYPE Dtype>
int OpFFT2d<Dtype>::checkTensorAttributes()
{
    if (validateRequiredOperands())
        return 1;

    if (validateRequiredRank(inputs[0]) || validateRequiredRank(inputs[1]) || validateRequiredRank(outputs[0]) ||
        validateRequiredRank(outputs[1]))
    {
        return 1;
    }

    if (inputs[0]->matchType(*outputs[0]) || inputs[1]->matchType(*outputs[1]) || inputs[0]->matchType(*inputs[1]))
    {
        printNodeValidationError("OpFFT2d: input and output tensor type mismatch");
        return 1;
    }

    in_real  = dynamic_cast<TosaReference::TensorTemplate<TIn>*>(inputs[0]);
    in_imag  = dynamic_cast<TosaReference::TensorTemplate<TIn>*>(inputs[1]);
    out_real = dynamic_cast<TosaReference::TensorTemplate<TOut>*>(outputs[0]);
    out_imag = dynamic_cast<TosaReference::TensorTemplate<TOut>*>(outputs[1]);

    ASSERT_MEM(in_real && in_imag && out_real && out_imag);

    std::string msg;
    if (check_fft_shape(in_real->getShape(), in_imag->getShape(), out_real->getShape(), out_imag->getShape(), msg))
    {
        msg = "OpFFT2d: " + msg;
        printNodeValidationError(msg.c_str());
        return 1;
    }

    return 0;
}

template <TOSA_REF_TYPE Dtype>
int OpFFT2d<Dtype>::eval()
{
    int in_real_batch  = this->in_real->getShape()[0];
    int in_real_height = this->in_real->getShape()[1];
    int in_real_width  = this->in_real->getShape()[2];

    int in_imag_batch  = this->in_imag->getShape()[0];
    int in_imag_height = this->in_imag->getShape()[1];
    int in_imag_width  = this->in_imag->getShape()[2];

    int out_real_batch  = this->out_real->getShape()[0];
    int out_real_height = this->out_real->getShape()[1];
    int out_real_width  = this->out_real->getShape()[2];

    int out_imag_batch  = this->out_imag->getShape()[0];
    int out_imag_height = this->out_imag->getShape()[1];
    int out_imag_width  = this->out_imag->getShape()[2];

    // Check Tosa Level
    auto tosa_level = g_func_config.tosa_level;
    LEVEL_CHECK(in_real_height <= tosa_level.MAX_KERNEL, "H should be smaller than or equal to MAX_KERNEL");
    LEVEL_CHECK(in_real_width <= tosa_level.MAX_KERNEL, "W should be smaller than or equal to MAX_KERNEL");

    DEBUG_INFO(OP, "perform OpFFT2d, input.shapes=[[%d,%d,%d],[%d,%d,%d]], output.shapes=[[%d,%d,%d],[%d,%d,%d]]",
               in_real_batch, in_real_height, in_real_width, in_imag_batch, in_imag_height, in_imag_width,
               out_real_batch, out_real_height, out_real_width, out_imag_batch, out_imag_height, out_imag_width);

    OutEigenType sum_real, sum_imag, sign_val = 1.0;
    OutEigenType a, a_cos, a_sin, v_ir;

    if (attribute->inverse())
    {
        sign_val = -1.0;
    }

    TIn in_real_val = this->in_real->getTensor();
    TIn in_imag_val = this->in_imag->getTensor();

    if (g_func_config.abs_mode)
    {
        // in abs_mode: take abs values of real and imag operands
        in_real_val = in_real_val.abs();
        in_imag_val = in_imag_val.abs();
    }

    for (int n = 0; n < in_real_batch; n++)
    {
        for (int oy = 0; oy < out_real_height; oy++)
        {
            for (int ox = 0; ox < out_real_width; ox++)
            {
                sum_real = 0.0;
                sum_imag = 0.0;
                for (int iy = 0; iy < in_real_height; iy++)
                {
                    for (int ix = 0; ix < in_real_width; ix++)
                    {
                        OutEigenType val_real = in_real_val(n, iy, ix);
                        OutEigenType val_imag = in_imag_val(n, iy, ix);
                        // Perform the periodic calculation in integer maths to keep
                        // the accuracy of the co-efficients similar for FP32 normal
                        // and FP64 precise mode
                        int32_t ay = (static_cast<int64_t>(iy) * static_cast<int64_t>(oy)) % in_real_height;
                        int32_t ax = (static_cast<int64_t>(ix) * static_cast<int64_t>(ox)) % in_real_width;

                        // Use explicit cast to ensure intermediate calculations are completed using OutEigenType
                        a = sign_val * 2 * M_PI *
                            ((OutEigenType)ay / in_real_height + (OutEigenType)ax / in_real_width);
                        // Calculate weight values
                        a_cos = cos(a);
                        a_sin = sin(a);
                        if (g_func_config.abs_mode)
                        {
                            // Bounded op - Use abs weight values
                            a_cos = std::abs(a_cos);
                            a_sin = std::abs(a_sin);
                            // Bounded op - Use abs real value for imaginary calc
                            v_ir = val_real;
                        }
                        else
                        {
                            // Normal op - Use negative real value for imaginary calc
                            v_ir = -val_real;
                        }
                        sum_real += val_real * a_cos + val_imag * a_sin;
                        sum_imag += v_ir * a_sin + val_imag * a_cos;
                    }
                }
                this->out_real->getTensor()(n, oy, ox) = sum_real;
                this->out_imag->getTensor()(n, oy, ox) = sum_imag;
            }
        }
    }

    return GraphNode::eval();
}

template <TOSA_REF_TYPE Dtype>
OpRFFT2d<Dtype>::OpRFFT2d(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_)
    : GraphNode(sgt_, Op_RFFT2D, id_)
{
    setRequiredOperands(1, 2);
    setRequiredRank(3, 3);

    INIT_ATTRIBUTE(RFFT);
}

template <TOSA_REF_TYPE Dtype>
OpRFFT2d<Dtype>::~OpRFFT2d()
{
    if (attribute)
        delete attribute;
}

template <TOSA_REF_TYPE Dtype>
int OpRFFT2d<Dtype>::checkTensorAttributes()
{
    if (validateRequiredOperands())
        return 1;

    if (validateRequiredRank(inputs[0]) || validateRequiredRank(outputs[0]) || validateRequiredRank(outputs[1]))
    {
        return 1;
    }

    if (inputs[0]->matchType(*outputs[0]) || inputs[0]->matchType(*outputs[1]))
    {
        printNodeValidationError("OpRFFT2d: input and output tensor type mismatch");
        return 1;
    }

    in       = dynamic_cast<TosaReference::TensorTemplate<TIn>*>(inputs[0]);
    out_real = dynamic_cast<TosaReference::TensorTemplate<TOut>*>(outputs[0]);
    out_imag = dynamic_cast<TosaReference::TensorTemplate<TOut>*>(outputs[1]);

    ASSERT_MEM(in && out_real && out_imag);

    std::string msg;
    if (check_fft_shape(in->getShape(), {}, out_real->getShape(), out_imag->getShape(), msg))
    {
        msg = "OpRFFT2d: " + msg;
        printNodeValidationError(msg.c_str());
        return 1;
    }

    return 0;
}

template <TOSA_REF_TYPE Dtype>
int OpRFFT2d<Dtype>::eval()
{
    int32_t in_batch  = in->getShape()[0];
    int32_t in_height = in->getShape()[1];
    int32_t in_width  = in->getShape()[2];

    int32_t out_real_batch  = out_real->getShape()[0];
    int32_t out_real_height = out_real->getShape()[1];
    int32_t out_real_width  = out_real->getShape()[2];

    int32_t out_imag_batch  = out_imag->getShape()[0];
    int32_t out_imag_height = out_imag->getShape()[1];
    int32_t out_imag_width  = out_imag->getShape()[2];

    int32_t half_in_height = in_height / 2;
    int32_t half_in_width  = in_width / 2;

    // Check Tosa Level
    auto tosa_level = g_func_config.tosa_level;
    LEVEL_CHECK(in_height <= tosa_level.MAX_KERNEL, "H should be smaller than or equal to MAX_KERNEL");
    LEVEL_CHECK(in_width <= tosa_level.MAX_KERNEL, "W should be smaller than or equal to MAX_KERNEL");

    DEBUG_INFO(OP,
               "perform OpRFFT2d, input.shape=[%d,%d,%d], output_real.shape=[%d,%d,%d], "
               "output_imag.shape=[%d,%d,%d]",
               in_batch, in_height, in_width, out_real_batch, out_real_height, out_real_width, out_imag_batch,
               out_imag_height, out_imag_width);

    OutEigenType sum_real, sum_imag;
    OutEigenType a, a_cos, a_sin, v_ir;

    TIn in_val = this->in->getTensor();

    if (g_func_config.abs_mode)
    {
        // in abs_mode: take abs values of in operand
        in_val = in_val.abs();
    }

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
                        OutEigenType val = in_val(n, iy, ix);
                        // Perform the periodic calculation in integer maths to keep
                        // the accuracy of the co-efficients similar for FP32 normal
                        // and FP64 precise mode
                        int32_t ay = (static_cast<int64_t>(iy) * static_cast<int64_t>(oy)) % in_height;
                        int32_t ax = (static_cast<int64_t>(ix) * static_cast<int64_t>(ox)) % in_width;

                        // Use explicit cast to ensure intermediate calculations are completed using OutEigenType
                        a = 2 * M_PI * ((OutEigenType)ay / in_height + (OutEigenType)ax / in_width);

                        // Calculate weight values (co-efficients)
                        a_cos = cos(a);
                        a_sin = sin(a);

                        if (g_func_config.abs_mode)
                        {
                            // Bounded op - Use abs weight values
                            a_cos = std::abs(a_cos);
                            a_sin = std::abs(a_sin);
                            // Bounded op - Use abs real value for imaginary calc
                            v_ir = val;
                        }
                        else
                        {
                            // Normal op - Use negative real value for imaginary calc
                            v_ir = -val;
                        }
                        sum_real += val * a_cos;
                        // Imaginary values with locations (0,0), (0,W/2), (H/2,0) and (H/2,W/2) are zero.
                        // But due to sin(M_PI) not returning 0 because of M_PI being approximate, only
                        // add to the imaginary sum when not processing these locations.
                        if ((in_height > 1 && (ay % (half_in_height)) > 0) ||
                            (in_width > 1 && (ax % (half_in_width)) > 0))
                        {
                            sum_imag += v_ir * a_sin;
                        }
                        else if (g_func_config.tosaExtraMultiplies())
                        {
                            sum_imag += v_ir * 0.0;
                        }
                    }
                }
                this->out_real->getTensor()(n, oy, ox) = sum_real;
                this->out_imag->getTensor()(n, oy, ox) = sum_imag;
            }
        }
    }

    return GraphNode::eval();
}

template <TOSA_REF_TYPE InDtype, TOSA_REF_TYPE WeightDtype, TOSA_REF_TYPE AccDtype, TOSA_REF_TYPE OutDtype>
OpTransposeConv2d<InDtype, WeightDtype, AccDtype, OutDtype>::OpTransposeConv2d(SubgraphTraverser* sgt_,
                                                                               TosaAttributeBase* attribute_,
                                                                               uint64_t id_)
    : GraphNode(sgt_, Op_TRANSPOSE_CONV2D, id_)
{
    setRequiredOperands(3, 1);
    setRequiredRank(4, 4);

    INIT_ATTRIBUTE(TransposeConv);
}

template <TOSA_REF_TYPE InDtype, TOSA_REF_TYPE WeightDtype, TOSA_REF_TYPE AccDtype, TOSA_REF_TYPE OutDtype>
OpTransposeConv2d<InDtype, WeightDtype, AccDtype, OutDtype>::~OpTransposeConv2d()
{
    if (attribute)
        delete attribute;
}

template <TOSA_REF_TYPE InDtype, TOSA_REF_TYPE WeightDtype, TOSA_REF_TYPE AccDtype, TOSA_REF_TYPE OutDtype>
int OpTransposeConv2d<InDtype, WeightDtype, AccDtype, OutDtype>::checkTensorAttributes()
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

    for (int32_t i : attribute->stride())
    {
        if (i < 1)
        {
            printNodeValidationError("OpTransposeConv2d: At least one stride is smaller than one");
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
        ERROR_IF(attribute->out_pad()[i] <= -(weight->getShape()[(i / 2) + 1]),
                 "OpTransposeConv2d: At least one out_pad value is larger than kernel size");
    }

    int32_t H = (IH - 1) * stride_y + out_pad_top + out_pad_bottom + kernel_h;
    int32_t W = (IW - 1) * stride_x + out_pad_left + out_pad_right + kernel_w;

    if ((OH != H) || (OW != W))
    {
        std::string msg = "OpTransposeConv2d: Mismatch between output shape provided and expected output shape (" +
                          std::to_string(H) + "," + std::to_string(W) + ")";
        printNodeValidationError(msg.c_str());
        return 1;
    }

    ERROR_IF(InDtype != TOSA_REF_TYPE_INT8 && attribute->input_zp() != 0,
             "OpTransposeConv2d: Input zeropoint must be zero for non int8_t data");
    ERROR_IF(WeightDtype != TOSA_REF_TYPE_INT8 && attribute->weight_zp() != 0,
             "OpTransposeConv2d: Weight zeropoint must be zero for non int8_t data");

    return 0;
}

template <TOSA_REF_TYPE InDtype, TOSA_REF_TYPE WeightDtype, TOSA_REF_TYPE AccDtype, TOSA_REF_TYPE OutDtype>
int OpTransposeConv2d<InDtype, WeightDtype, AccDtype, OutDtype>::eval()
{
    int in_batch    = this->input->getShape()[0];
    int in_height   = this->input->getShape()[1];
    int in_width    = this->input->getShape()[2];
    int in_channels = this->input->getShape()[3];

    int k_out_channels = this->weight->getShape()[0];
    int k_height       = this->weight->getShape()[1];
    int k_width        = this->weight->getShape()[2];
    int k_in_channels  = this->weight->getShape()[3];

    int b_out_channels = this->bias->getShape()[0];

    int out_batch    = this->output->getShape()[0];
    int out_height   = this->output->getShape()[1];
    int out_width    = this->output->getShape()[2];
    int out_channels = this->output->getShape()[3];

    int out_pad_top    = this->attribute->out_pad()[0];
    int out_pad_bottom = this->attribute->out_pad()[1];
    int out_pad_left   = this->attribute->out_pad()[2];
    int out_pad_right  = this->attribute->out_pad()[3];

    int stride_y = this->attribute->stride()[0];
    int stride_x = this->attribute->stride()[1];

    ERROR_IF(in_batch != out_batch, "OpTransposeConv2d: tensor batch mismatch %d != %d", in_batch, out_batch);
    ERROR_IF(k_in_channels != in_channels, "OpTransposeConv2d: tensor input channel mismatch %d != %d", k_in_channels,
             in_channels);
    ERROR_IF(k_out_channels != out_channels, "OpTransposeConv2d: tensor output channel mismatch %d != %d",
             k_out_channels, out_channels);
    ERROR_IF(b_out_channels != out_channels && b_out_channels != 1,
             "OpTransposeConv2d: bias channels mismatch %d != %d", b_out_channels, out_channels);

    // Check Tosa Level
    auto tosa_level = g_func_config.tosa_level;
    LEVEL_CHECK(k_height <= tosa_level.MAX_KERNEL, "KH should be smaller than or equal to MAX_KERNEL");
    LEVEL_CHECK(k_width <= tosa_level.MAX_KERNEL, "KW should be smaller than or equal to MAX_KERNEL");
    LEVEL_CHECK(out_pad_top <= tosa_level.MAX_KERNEL, "out_pad_top should be smaller than or equal to MAX_KERNEL");
    LEVEL_CHECK(out_pad_bottom <= tosa_level.MAX_KERNEL,
                "out_pad_bottom should be smaller than or equal to MAX_KERNEL");
    LEVEL_CHECK(out_pad_left <= tosa_level.MAX_KERNEL, "out_pad_left should be smaller than or equal to MAX_KERNEL");
    LEVEL_CHECK(out_pad_right <= tosa_level.MAX_KERNEL, "out_pad_right should be smaller than or equal to MAX_KERNEL");
    LEVEL_CHECK(stride_y <= tosa_level.MAX_STRIDE, "stride_y should be smaller than or equal to MAX_STRIDE");
    LEVEL_CHECK(stride_x <= tosa_level.MAX_STRIDE, "stride_x should be smaller than or equal to MAX_STRIDE");

    DEBUG_INFO(OP,
               "perform OpTransposeConv2d, input.shape=[%d,%d,%d,%d], weight.shape=[%d,%d,%d,%d], "
               "output.shape=[%d,%d,%d,%d], stride=[%d,%d], out_pad=[%d,%d,%d,%d]",
               in_batch, in_height, in_width, in_channels, k_height, k_width, k_out_channels, k_in_channels, out_batch,
               out_height, out_width, out_channels, stride_y, stride_x, out_pad_top, out_pad_bottom, out_pad_left,
               out_pad_right);

    TIn input_val      = this->input->getTensor();
    TWeight weight_val = this->weight->getTensor();
    if (InDtype == TOSA_REF_TYPE_INT8 || WeightDtype == TOSA_REF_TYPE_INT8)
    {
        input_val  = input_val - (InEigenType)attribute->input_zp();
        weight_val = weight_val - (WeightEigenType)attribute->weight_zp();
    }

    TBias bias_val = this->bias->getTensor();

    if (g_func_config.abs_mode)
    {
        // in abs_mode: take abs values of conv operands
        input_val  = input_val.abs();
        weight_val = weight_val.abs();
        bias_val   = bias_val.abs();

        if (!this->attribute->local_bound())
        {
            Eigen::Tensor<InEigenType, 0> input_abs_max = input_val.maximum();
            input_val.setConstant(input_abs_max(0));
        }
    }

    Eigen::array<Eigen::Index, 4> reshape_dim;
    reshape_dim.fill(1);
    reshape_dim[3] = b_out_channels;

    Eigen::array<Eigen::Index, 4> bcast;
    bcast[0] = out_batch;
    bcast[1] = out_height;
    bcast[2] = out_width;
    bcast[3] = (b_out_channels == 1) ? out_channels : 1;

    // initialize with bias
    this->output->getTensor() = bias_val.reshape(reshape_dim).broadcast(bcast);

    TAcc acc_tensor = this->output->getTensor().template cast<AccEigenType>();
    acc_tensor.setZero();

    // reference implementation from: tensorflow/tensorflow/lite/kernels/internal/reference/reference_ops.h
    for (int ob = 0; ob < out_batch; ob++)
    {
        for (int oy = 0; oy < out_height; oy++)
        {
            for (int ox = 0; ox < out_width; ox++)
            {
                int iy = oy - out_pad_top;
                int ix = ox - out_pad_left;
                for (int oc = 0; oc < out_channels; oc++)
                {
                    for (int ky = 0; ky < k_height; ky++)
                    {
                        for (int kx = 0; kx < k_width; kx++)
                        {
                            int y         = iy - ky;
                            int x         = ix - kx;
                            bool in_scope = (0 <= y && y < (in_height * stride_y)) &&
                                            (0 <= x && x < (in_width * stride_x)) && ((y % stride_y) == 0) &&
                                            ((x % stride_x) == 0);
                            if (!in_scope && !g_func_config.tosaExtraMultiplies())
                            {
                                // no need to do multiply and accumulate
                                continue;
                            }

                            for (int ic = 0; ic < in_channels; ic++)
                            {
                                AccEigenType value(0.0);
                                if (in_scope)
                                {
                                    value = (AccEigenType)input_val(ob, y / stride_y, x / stride_x, ic);
                                }
                                acc_tensor(ob, oy, ox, oc) += value * (AccEigenType)weight_val(oc, ky, kx, ic);
                            }
                        }
                    }
                }
            }
        }
    }

    this->output->getTensor() += acc_tensor.template cast<OutEigenType>();

    if (OutDtype == TOSA_REF_TYPE_INT48)
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
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpArgMax, FP64);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpArgMax, FP8E4M3);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpArgMax, FP8E5M2);

DEF_INSTANTIATE_TWO_TYPE(OpAvgPool2d, FP16, FP16);
DEF_INSTANTIATE_TWO_TYPE(OpAvgPool2d, FP16, FP32);
DEF_INSTANTIATE_TWO_TYPE(OpAvgPool2d, BF16, FP32);
DEF_INSTANTIATE_TWO_TYPE(OpAvgPool2d, FP32, FP32);
DEF_INSTANTIATE_TWO_TYPE(OpAvgPool2d, INT8, INT32);
DEF_INSTANTIATE_TWO_TYPE(OpAvgPool2d, INT16, INT32);
DEF_INSTANTIATE_TWO_TYPE(OpAvgPool2d, FP64, FP64);
DEF_INSTANTIATE_TWO_TYPE(OpAvgPool2d, FP8E4M3, FP16);
DEF_INSTANTIATE_TWO_TYPE(OpAvgPool2d, FP8E5M2, FP16);

// [in_t, weight_t, acc_t, out_t]
DEF_INSTANTIATE_FOUR_TYPE(OpConv2d, FP16, FP16, FP16, FP16);
DEF_INSTANTIATE_FOUR_TYPE(OpConv2d, FP16, FP16, FP32, FP16);
DEF_INSTANTIATE_FOUR_TYPE(OpConv2d, BF16, BF16, FP32, BF16);
DEF_INSTANTIATE_FOUR_TYPE(OpConv2d, FP32, FP32, FP32, FP32);
DEF_INSTANTIATE_FOUR_TYPE(OpConv2d, INT8, INT4, INT32, INT32);
DEF_INSTANTIATE_FOUR_TYPE(OpConv2d, INT8, INT8, INT32, INT32);
DEF_INSTANTIATE_FOUR_TYPE(OpConv2d, INT16, INT8, INT48, INT48);
DEF_INSTANTIATE_FOUR_TYPE(OpConv2d, FP64, FP64, FP64, FP64);
DEF_INSTANTIATE_FOUR_TYPE(OpConv2d, FP8E4M3, FP8E4M3, FP16, FP16);
DEF_INSTANTIATE_FOUR_TYPE(OpConv2d, FP8E5M2, FP8E5M2, FP16, FP16);

DEF_INSTANTIATE_FOUR_TYPE(OpConv3d, FP16, FP16, FP16, FP16);
DEF_INSTANTIATE_FOUR_TYPE(OpConv3d, FP16, FP16, FP32, FP16);
DEF_INSTANTIATE_FOUR_TYPE(OpConv3d, BF16, BF16, FP32, BF16);
DEF_INSTANTIATE_FOUR_TYPE(OpConv3d, FP32, FP32, FP32, FP32);
DEF_INSTANTIATE_FOUR_TYPE(OpConv3d, INT8, INT4, INT32, INT32);
DEF_INSTANTIATE_FOUR_TYPE(OpConv3d, INT8, INT8, INT32, INT32);
DEF_INSTANTIATE_FOUR_TYPE(OpConv3d, INT16, INT8, INT48, INT48);
DEF_INSTANTIATE_FOUR_TYPE(OpConv3d, FP64, FP64, FP64, FP64);
DEF_INSTANTIATE_FOUR_TYPE(OpConv3d, FP8E4M3, FP8E4M3, FP16, FP16);
DEF_INSTANTIATE_FOUR_TYPE(OpConv3d, FP8E5M2, FP8E5M2, FP16, FP16);

DEF_INSTANTIATE_FOUR_TYPE(OpDepthwiseConv2d, FP16, FP16, FP16, FP16);
DEF_INSTANTIATE_FOUR_TYPE(OpDepthwiseConv2d, FP16, FP16, FP32, FP16);
DEF_INSTANTIATE_FOUR_TYPE(OpDepthwiseConv2d, BF16, BF16, FP32, BF16);
DEF_INSTANTIATE_FOUR_TYPE(OpDepthwiseConv2d, FP32, FP32, FP32, FP32);
DEF_INSTANTIATE_FOUR_TYPE(OpDepthwiseConv2d, INT8, INT4, INT32, INT32);
DEF_INSTANTIATE_FOUR_TYPE(OpDepthwiseConv2d, INT8, INT8, INT32, INT32);
DEF_INSTANTIATE_FOUR_TYPE(OpDepthwiseConv2d, INT16, INT8, INT48, INT48);
DEF_INSTANTIATE_FOUR_TYPE(OpDepthwiseConv2d, FP64, FP64, FP64, FP64);
DEF_INSTANTIATE_FOUR_TYPE(OpDepthwiseConv2d, FP8E4M3, FP8E4M3, FP16, FP16);
DEF_INSTANTIATE_FOUR_TYPE(OpDepthwiseConv2d, FP8E5M2, FP8E5M2, FP16, FP16);

DEF_INSTANTIATE_ONE_TYPE(OpFFT2d, FP32);
DEF_INSTANTIATE_ONE_TYPE(OpFFT2d, FP64);

DEF_INSTANTIATE_THREE_TYPE(OpMatMul, INT8, INT8, INT32);
DEF_INSTANTIATE_THREE_TYPE(OpMatMul, INT16, INT16, INT48);
DEF_INSTANTIATE_THREE_TYPE(OpMatMul, FP16, FP16, FP16);
DEF_INSTANTIATE_THREE_TYPE(OpMatMul, FP16, FP16, FP32);
DEF_INSTANTIATE_THREE_TYPE(OpMatMul, BF16, BF16, FP32);
DEF_INSTANTIATE_THREE_TYPE(OpMatMul, FP32, FP32, FP32);
DEF_INSTANTIATE_THREE_TYPE(OpMatMul, FP64, FP64, FP64);
DEF_INSTANTIATE_THREE_TYPE(OpMatMul, FP8E4M3, FP8E4M3, FP16);
DEF_INSTANTIATE_THREE_TYPE(OpMatMul, FP8E5M2, FP8E5M2, FP16);
DEF_INSTANTIATE_THREE_TYPE(OpMatMul, FP8E4M3, FP8E4M3, FP32);
DEF_INSTANTIATE_THREE_TYPE(OpMatMul, FP8E5M2, FP8E5M2, FP32);
DEF_INSTANTIATE_THREE_TYPE(OpMatMul, FP8E4M3, FP8E5M2, FP16);
DEF_INSTANTIATE_THREE_TYPE(OpMatMul, FP8E5M2, FP8E4M3, FP16);
DEF_INSTANTIATE_THREE_TYPE(OpMatMul, FP8E4M3, FP8E5M2, FP32);
DEF_INSTANTIATE_THREE_TYPE(OpMatMul, FP8E5M2, FP8E4M3, FP32);

DEF_INSTANTIATE_ONE_TYPE(OpMaxPool2d, FP16);
DEF_INSTANTIATE_ONE_TYPE(OpMaxPool2d, BF16);
DEF_INSTANTIATE_ONE_TYPE(OpMaxPool2d, FP32);
DEF_INSTANTIATE_ONE_TYPE(OpMaxPool2d, INT8);
DEF_INSTANTIATE_ONE_TYPE(OpMaxPool2d, INT16);
DEF_INSTANTIATE_ONE_TYPE(OpMaxPool2d, FP64);
DEF_INSTANTIATE_ONE_TYPE(OpMaxPool2d, FP8E4M3);
DEF_INSTANTIATE_ONE_TYPE(OpMaxPool2d, FP8E5M2);

DEF_INSTANTIATE_ONE_TYPE(OpRFFT2d, FP32);
DEF_INSTANTIATE_ONE_TYPE(OpRFFT2d, FP64);

// [in_t, weight_t, acc_t, out_t]
DEF_INSTANTIATE_FOUR_TYPE(OpTransposeConv2d, FP16, FP16, FP16, FP16);
DEF_INSTANTIATE_FOUR_TYPE(OpTransposeConv2d, FP16, FP16, FP32, FP16);
DEF_INSTANTIATE_FOUR_TYPE(OpTransposeConv2d, BF16, BF16, FP32, BF16);
DEF_INSTANTIATE_FOUR_TYPE(OpTransposeConv2d, FP32, FP32, FP32, FP32);
DEF_INSTANTIATE_FOUR_TYPE(OpTransposeConv2d, INT8, INT4, INT32, INT32);
DEF_INSTANTIATE_FOUR_TYPE(OpTransposeConv2d, INT8, INT8, INT32, INT32);
DEF_INSTANTIATE_FOUR_TYPE(OpTransposeConv2d, INT16, INT8, INT48, INT48);
DEF_INSTANTIATE_FOUR_TYPE(OpTransposeConv2d, FP64, FP64, FP64, FP64);
DEF_INSTANTIATE_FOUR_TYPE(OpTransposeConv2d, FP8E4M3, FP8E4M3, FP16, FP16);
DEF_INSTANTIATE_FOUR_TYPE(OpTransposeConv2d, FP8E5M2, FP8E5M2, FP16, FP16);

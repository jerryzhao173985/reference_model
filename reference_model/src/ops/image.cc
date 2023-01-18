
// Copyright (c) 2020-2022, ARM Limited.
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

#include "image.h"
#include "arith_util.h"
#include "half.hpp"

#include <type_traits>

using namespace TosaReference;
using namespace Eigen;
using namespace tosa;

template <DType InDtype, DType OutDtype, typename resize_t>
OpResize<InDtype, OutDtype, resize_t>::OpResize(SubgraphTraverser* sgt_,
                                                TosaAttributeBase* attribute_,
                                                uint64_t id_)
    : GraphNode(sgt_, Op_RESIZE, id_)
{
    setRequiredOperands(1, 1);
    setRequiredRank(4, 4);

    INIT_ATTRIBUTE(Resize);
}

template <DType InDtype, DType OutDtype, typename resize_t>
OpResize<InDtype, OutDtype, resize_t>::~OpResize()
{
    if (attribute)
        delete attribute;
}

template <DType InDtype, DType OutDtype, typename resize_t>
int OpResize<InDtype, OutDtype, resize_t>::checkTensorAttributes()
{
    if (validateRequiredOperands())
        return 1;

    if (validateRequiredRank(inputs[0]) || validateRequiredRank(outputs[0]))
        return 1;

    if (this->attribute->scale().size() != 4)
    {
        printNodeValidationError("OpResize: illegal size for attribute scale");
        return 1;
    }

    scale  = this->attribute->scale();
    offset = this->attribute->offset();
    border = this->attribute->border();
    mode   = this->attribute->mode();

    if (this->mode == ResizeMode_BILINEAR)
    {
        if (OutDtype != DType_INT32 && OutDtype != DType_INT48 && OutDtype != DType_FP32 && OutDtype != DType_FP16 && OutDtype != DType_BF16)
        {
            printNodeValidationError("OpResize: invalid data type for BILINEAR");
            return 1;
        }
    }
    else
    {
        if (OutDtype != DType_INT8 && OutDtype != DType_INT16 && OutDtype != DType_FP32 && OutDtype != DType_FP16 && OutDtype != DType_BF16)
        {
            printNodeValidationError("OpResize: invalid data type for NEAREST");
            return 1;
        }
    }

    in  = dynamic_cast<TosaReference::TensorTemplate<TIn>*>(inputs[0]);
    out = dynamic_cast<TosaReference::TensorTemplate<TOut>*>(outputs[0]);

    ASSERT_MEM(in && out);

    return 0;
}

template <DType InDtype, DType OutDtype, typename resize_t>
int OpResize<InDtype, OutDtype, resize_t>::eval()
{
    int in_batch    = in->getShape()[0];
    int in_height   = in->getShape()[1];
    int in_width    = in->getShape()[2];
    int in_channels = in->getShape()[3];

    int out_batch    = out->getShape()[0];
    int out_height   = out->getShape()[1];
    int out_width    = out->getShape()[2];
    int out_channels = out->getShape()[3];

    int16_t scale_y_n = scale[0];
    int16_t scale_y_d = scale[1];
    int16_t scale_x_n = scale[2];
    int16_t scale_x_d = scale[3];

    int16_t offset_y = offset[0];
    int16_t offset_x = offset[1];

    int16_t border_y = border[0];
    int16_t border_x = border[1];

    ERROR_IF(std::max<int>({ in_height, in_width, out_height, out_width }) >= 16384,
             "OpResize: exceeds maximum dimension");
    ERROR_IF(in_batch != out_batch, "OpResize: output tensor batch mismatch");
    ERROR_IF(in_channels != out_channels, "OpResize: output tensor channel mismatch");
    ERROR_IF(scale_y_n <= 0 || scale_y_d <= 0 || scale_x_n <= 0 || scale_x_d <= 0,
             "OpResize: attribute scale must not be negative");
    // If data type is int8_t then ensure that an int32_t accumulator can be used.
    ERROR_IF(scale_y_n > (1 << 11) || scale_x_n > (1 << 11), "OpResize: invalid attribute scale");
    // Set a consistent lower limit of 1/16 downscale to simplify implementations
    ERROR_IF((scale_y_d >= 16 * scale_y_n) || (scale_x_d >= 16 * scale_x_n), "OpResize: invalid attribute scale");
    ERROR_IF((offset_y < -scale_y_n) || (offset_y >= 16 * scale_y_n),
             "OpResize: invalid attribute offset height dimension");
    ERROR_IF((offset_x < -scale_x_n) || (offset_x >= 16 * scale_x_n),
             "OpResize: invalid attribute offset width dimension");
    ERROR_IF((border_y < -16 * scale_y_n || border_y >= scale_y_n),
             "OpResize: invalid attribute border height dimension");
    ERROR_IF((border_x < -16 * scale_x_n || border_x >= scale_x_n),
             "OpResize: invalid attribute border width dimension");

    int32_t res_height = 0;
    int32_t res_width = 0;

    if (idiv_check((in_height - 1) * scale_y_n - offset_y + border_y, scale_y_d, res_height))
       return 1;

    if (idiv_check((in_width - 1) * scale_x_n - offset_x + border_x, scale_x_d, res_width))
       return 1;

    ERROR_IF(out_height != res_height + 1,
             "OpResize: mismatch between output height dimension provided and expected shape");
    ERROR_IF(out_width != res_width + 1,
             "OpResize: mismatch between output width dimension provided and expected shape");

    for (int b = 0; b < out_batch; b++)
        for (int c = 0; c < out_channels; c++)
            for (int oy = 0; oy < out_height; oy++)
                for (int ox = 0; ox < out_width; ox++)
                {
                    int32_t y = oy * scale_y_d + offset_y;
                    int32_t x = ox * scale_x_d + offset_x;

                    float fy = static_cast<float>(y) / static_cast<float>(scale_y_n);
                    float fx = static_cast<float>(x) / static_cast<float>(scale_x_n);

                    int32_t iy = floor(fy);
                    int32_t ix = floor(fx);

                    resize_t dy;
                    resize_t dx;
                    if (std::is_floating_point<resize_t>::value || (typeid(resize_t) == typeid(Eigen::bfloat16)) ||
                        (typeid(resize_t) == typeid(half_float::half)))
                    {
                        dy = (resize_t)(fy - iy);
                        dx = (resize_t)(fx - ix);
                    }
                    else
                    {
                        dy = (resize_t)(y - (iy * scale_y_n));
                        dx = (resize_t)(x - (ix * scale_x_n));
                    }

                    int32_t iy0 = MAX(iy, 0);
                    int32_t iy1 = MIN(iy + 1, in_height - 1);
                    int32_t ix0 = MAX(ix, 0);
                    int32_t ix1 = MIN(ix + 1, in_width - 1);

                    OutEigenType acc;
                    if (mode == ResizeMode_BILINEAR)
                    {
                        InEigenType v00 = in->getTensor()(b, iy0, ix0, c);
                        InEigenType v01 = in->getTensor()(b, iy0, ix1, c);
                        InEigenType v10 = in->getTensor()(b, iy1, ix0, c);
                        InEigenType v11 = in->getTensor()(b, iy1, ix1, c);

                        if (std::is_floating_point<resize_t>::value)
                        {
                            acc = (OutEigenType)v00 * (1.0 - dy) * (1.0 - dx);
                            acc += (OutEigenType)v01 * (1.0 - dy) * dx;
                            acc += (OutEigenType)v10 * dy * (1.0 - dx);
                            acc += (OutEigenType)v11 * dy * dx;
                        }
                        else if ((typeid(resize_t) == typeid(Eigen::bfloat16)) ||
                                 (typeid(resize_t) == typeid(half_float::half)))
                        {
                            resize_t f16_acc;
                            f16_acc = (resize_t)v00 * (resize_t)(1.0 - dy) * (resize_t)(1.0 - dx);
                            f16_acc += (resize_t)v01 * (resize_t)(1.0 - dy) * (resize_t)dx;
                            f16_acc += (resize_t)v10 * (resize_t)dy * (resize_t)(1.0 - dx);
                            f16_acc += (resize_t)v11 * (resize_t)dy * (resize_t)dx;
                            acc = (float)f16_acc;
                        }
                        else
                        {
                            acc = (OutEigenType)v00 * (scale_y_n - dy) * (scale_x_n - dx);
                            acc += (OutEigenType)v01 * (scale_y_n - dy) * dx;
                            acc += (OutEigenType)v10 * dy * (scale_x_n - dx);
                            acc += (OutEigenType)v11 * dy * dx;
                        }
                    }
                    else
                    {
                        ASSERT_MSG(mode == ResizeMode_NEAREST, "OpResize: invalid mode");
                        if (std::is_floating_point<resize_t>::value || (typeid(resize_t) == typeid(Eigen::bfloat16)) ||
                            (typeid(resize_t) == typeid(half_float::half)))
                        {
                            iy = (dy >= 0.5) ? iy1 : iy0;
                            ix = (dx >= 0.5) ? ix1 : ix0;
                        }
                        else
                        {
                            iy = (2 * dy >= scale_y_n) ? iy1 : iy0;
                            ix = (2 * dx >= scale_x_n) ? ix1 : ix0;
                        }
                        acc = in->getTensor()(b, iy, ix, c);
                    }
                    if ((typeid(resize_t) == typeid(Eigen::bfloat16))) {
                        ASSERT_MSG(checkValidBFloat(acc), "Resize accumulator float value is not a valid bfloat16 value.");
                    }
                    out->getTensor()(b, oy, ox, c) = acc;
                }

    return GraphNode::eval();
}

// template explicit instantiation
DEF_INSTANTIATE_THREE_TYPE_RESIZE(OpResize, INT8, INT32, int16_t);
DEF_INSTANTIATE_THREE_TYPE_RESIZE(OpResize, INT8, INT8, int16_t);
DEF_INSTANTIATE_THREE_TYPE_RESIZE(OpResize, INT16, INT48, int16_t);
DEF_INSTANTIATE_THREE_TYPE_RESIZE(OpResize, INT16, INT16, int16_t);
DEF_INSTANTIATE_THREE_TYPE_RESIZE(OpResize, FP16, FP16, half_float::half);
DEF_INSTANTIATE_THREE_TYPE_RESIZE(OpResize, BF16, BF16, Eigen::bfloat16);
DEF_INSTANTIATE_THREE_TYPE_RESIZE(OpResize, FP32, FP32, float);

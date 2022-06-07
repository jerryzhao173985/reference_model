
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

#include "image.h"
#include "arith_util.h"

using namespace TosaReference;
using namespace Eigen;
using namespace tosa;

template <DType InDtype, DType OutDtype>
OpResize<InDtype, OutDtype>::OpResize(SubgraphTraverser* sgt_,
                                      TosaAttributeBase* attribute_,
                                      uint64_t id_)
    : GraphNode(sgt_, Op_RESIZE, id_)
{
    setRequiredOperands(1, 1);
    setRequiredRank(4, 4);

    INIT_ATTRIBUTE(Resize);
}

template <DType InDtype, DType OutDtype>
OpResize<InDtype, OutDtype>::~OpResize()
{
    if (attribute)
        delete attribute;
}

template <DType InDtype, DType OutDtype>
int OpResize<InDtype, OutDtype>::checkTensorAttributes()
{
    if (validateRequiredOperands())
        return 1;

    if (validateRequiredRank(inputs[0]) || validateRequiredRank(outputs[0]))
        return 1;

    output_size = this->attribute->output_size();
    stride      = this->attribute->stride();
    offset      = this->attribute->offset();
    shift       = this->attribute->shift();
    stride_fp   = this->attribute->stride_fp();
    offset_fp   = this->attribute->offset_fp();
    mode        = this->attribute->mode();

    int output_height = outputs[0]->getShape()[1];
    int output_width  = outputs[0]->getShape()[2];

    if (this->mode == ResizeMode_BILINEAR)
    {
        if (OutDtype != DType_INT32 && OutDtype != DType_INT48 && OutDtype != DType_FLOAT)
        {
            printNodeValidationError("OpResize: invalid data type for BILINEAR");
            return 1;
        }
    }
    else
    {
        if (OutDtype != DType_INT8 && OutDtype != DType_INT16 && OutDtype != DType_FLOAT)
        {
            printNodeValidationError("OpResize: invalid data type for NEAREST");
            return 1;
        }
    }

    if (output_size[0] != output_height || output_size[1] != output_width)
    {
        printNodeValidationError("OpResize: attribute output_size doesn't match output [height, width]");
        return 1;
    }

    in  = dynamic_cast<TosaReference::TensorTemplate<TIn>*>(inputs[0]);
    out = dynamic_cast<TosaReference::TensorTemplate<TOut>*>(outputs[0]);

    ASSERT_MEM(in && out);

    return 0;
}

template <DType InDtype, DType OutDtype>
int OpResize<InDtype, OutDtype>::eval()
{
    int in_batch    = in->getShape()[0];
    int in_height   = in->getShape()[1];
    int in_width    = in->getShape()[2];
    int in_channels = in->getShape()[3];

    int out_batch    = out->getShape()[0];
    int out_height   = out->getShape()[1];
    int out_width    = out->getShape()[2];
    int out_channels = out->getShape()[3];

    ERROR_IF(std::max<int>({ in_height, in_width, out_height, out_width }) >= 16384,
             "OpResize: exceeds maximum dimension");
    ERROR_IF(shift < 1 || shift > 11, "OpResize: attribute shift should be within [1, 11]");
    ERROR_IF(stride[0] <= 0 || stride[0] >= (16 << shift), "OpResize: invalid attribute stride_x");
    ERROR_IF(stride[1] <= 0 || stride[1] >= (16 << shift), "OpResize: invalid attribute stride_y");
    ERROR_IF(offset[0] <= (-16 << shift) || offset[0] >= (16 << shift), "OpResize: invalid attribute offset_x");
    ERROR_IF(offset[1] <= (-16 << shift) || offset[1] >= (16 << shift), "OpResize: invalid attribute offset_y");
    ERROR_IF(in_batch != out_batch, "OpResize: output tensor batch mismatch");
    ERROR_IF(in_channels != out_channels, "OpResize: output tensor channel mismatch");

    for (int b = 0; b < out_batch; b++)
        for (int c = 0; c < out_channels; c++)
            for (int oy = 0; oy < out_height; oy++)
                for (int ox = 0; ox < out_width; ox++)
                {
                    int32_t y = oy * stride[0] + offset[0];
                    int32_t x = ox * stride[1] + offset[1];

                    int32_t iy = y >> shift;
                    int32_t dy = y - (iy << shift);
                    int32_t ix = x >> shift;
                    int32_t dx = x - (ix << shift);

                    int32_t iy0 = MAX(iy, 0);
                    int32_t iy1 = MIN(iy + 1, in_height - 1);
                    int32_t ix0 = MAX(ix, 0);
                    int32_t ix1 = MIN(ix + 1, in_width - 1);

                    REQUIRE(iy0 <= iy1 && ix0 <= ix1, "OpResize: invalid index (iy0, iy1, ix0, ix1)=(%d,%d,%d,%d)", iy0,
                            iy1, ix0, ix1);

                    OutEigenType acc;
                    if (mode == ResizeMode_BILINEAR)
                    {
                        InEigenType v00 = in->getTensor()(b, iy0, ix0, c);
                        InEigenType v01 = in->getTensor()(b, iy0, ix1, c);
                        InEigenType v10 = in->getTensor()(b, iy1, ix0, c);
                        InEigenType v11 = in->getTensor()(b, iy1, ix1, c);

                        acc = (OutEigenType)v00 * ((1 << shift) - dy) * ((1 << shift) - dx);
                        acc = acc + (OutEigenType)v01 * ((1 << shift) - dy) * dx;
                        acc = acc + (OutEigenType)v10 * dy * ((1 << shift) - dx);
                        acc = acc + (OutEigenType)v11 * dy * dx;
                    }
                    else
                    {
                        iy  = (dy >> (shift - 1)) != 0 ? iy1 : iy0;
                        ix  = (dx >> (shift - 1)) != 0 ? ix1 : ix0;
                        acc = in->getTensor()(b, iy, ix, c);
                    }

                    out->getTensor()(b, oy, ox, c) = acc;
                }

    return GraphNode::eval();
}

template <>
int OpResize<DType_FLOAT, DType_FLOAT>::eval()
{
    int in_batch    = in->getShape()[0];
    int in_height   = in->getShape()[1];
    int in_width    = in->getShape()[2];
    int in_channels = in->getShape()[3];

    int out_batch    = out->getShape()[0];
    int out_height   = out->getShape()[1];
    int out_width    = out->getShape()[2];
    int out_channels = out->getShape()[3];

    ERROR_IF(std::max<int>({ in_height, in_width, out_height, out_width }) >= 16384,
             "OpResize: exceeds maximum dimension");
    ERROR_IF(shift != 0, "OpResize: float mode must have 0 shift");
    ERROR_IF(stride_fp[0] <= 0.0f || stride_fp[1] <= 0.0f, "OpResize: invalid attribute stride");
    ERROR_IF(stride_fp[0] > in_height || stride_fp[1] > in_width, "OpResize: stride larger than dimension");
    ERROR_IF(in_batch != out_batch, "OpResize: output tensor batch mismatch");
    ERROR_IF(in_channels != out_channels, "OpResize: output tensor channel mismatch");

    for (int b = 0; b < out_batch; b++)
        for (int c = 0; c < out_channels; c++)
            for (int oy = 0; oy < out_height; oy++)
                for (int ox = 0; ox < out_width; ox++)
                {
                    float y = oy * stride_fp[0] + offset_fp[0];
                    float x = ox * stride_fp[1] + offset_fp[1];

                    int32_t iy = static_cast<int32_t>(std::floor(y));
                    float dy   = y - static_cast<float>(iy);
                    int32_t ix = static_cast<int32_t>(std::floor(x));
                    float dx   = x - static_cast<float>(ix);

                    int32_t iy0 = MAX(iy, 0);
                    int32_t iy1 = MIN(iy + 1, in_height - 1);
                    int32_t ix0 = MAX(ix, 0);
                    int32_t ix1 = MIN(ix + 1, in_width - 1);

                    REQUIRE(iy0 <= iy1 && ix0 <= ix1, "OpResize: invalid index (iy0, iy1, ix0, ix1)=(%d,%d,%d,%d)", iy0,
                            iy1, ix0, ix1);

                    OutEigenType acc;
                    if (mode == ResizeMode_BILINEAR)
                    {
                        InEigenType v00 = in->getTensor()(b, iy0, ix0, c);
                        InEigenType v01 = in->getTensor()(b, iy0, ix1, c);
                        InEigenType v10 = in->getTensor()(b, iy1, ix0, c);
                        InEigenType v11 = in->getTensor()(b, iy1, ix1, c);

                        acc = (OutEigenType)v00 * (1.0 - dy) * (1.0 - dx);
                        acc = acc + (OutEigenType)v01 * (1.0 - dy) * dx;
                        acc = acc + (OutEigenType)v10 * dy * (1.0 - dx);
                        acc = acc + (OutEigenType)v11 * dy * dx;
                    }
                    else
                    {
                        iy  = (dy >= 0.5) ? iy1 : iy0;
                        ix  = (dx >= 0.5) ? ix1 : ix0;
                        acc = in->getTensor()(b, iy, ix, c);
                    }

                    out->getTensor()(b, oy, ox, c) = acc;
                }

    return GraphNode::eval();
}

// template explicit instantiation
DEF_INSTANTIATE_TWO_TYPE(OpResize, INT8, INT32);
DEF_INSTANTIATE_TWO_TYPE(OpResize, INT8, INT8);
DEF_INSTANTIATE_TWO_TYPE(OpResize, INT16, INT48);
DEF_INSTANTIATE_TWO_TYPE(OpResize, INT16, INT16);
DEF_INSTANTIATE_TWO_TYPE(OpResize, FLOAT, FLOAT);

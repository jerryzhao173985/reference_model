// Copyright (c) 2023-2024, ARM Limited.
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

#include "generate_dot_product.h"
#include "half.hpp"

namespace
{
//---------------------------------------------------------------------------//
//                              MatMul                                       //
//---------------------------------------------------------------------------//

template <typename DataType>
void generateMatMulA(const TosaReference::GenerateConfig& cfg,
                     TosaReference::IDotProductGenerator& generator,
                     DataType* data,
                     size_t size)
{
    const uint32_t T = cfg.shape[0] * cfg.shape[1] * cfg.shape[2];
    const uint32_t C = cfg.shape[2];

    for (uint32_t t = 0; t < T; ++t)
    {
        data[t] = static_cast<DataType>(generator(t % C));    // k = c
    }
}

template <typename DataType>
void generateMatMulB(const TosaReference::GenerateConfig& cfg,
                     TosaReference::IDotProductGenerator& generator,
                     DataType* data,
                     size_t size)
{
    const uint32_t T = cfg.shape[0] * cfg.shape[1] * cfg.shape[2];
    const uint32_t C = cfg.shape[1];
    const uint32_t W = cfg.shape[2];

    for (uint32_t t = 0; t < T; ++t)
    {
        data[t] = static_cast<DataType>(generator((t / W) % C));    // k = c
    }
}

bool generateMatMul(const TosaReference::GenerateConfig& cfg,
                    TosaReference::IDotProductGenerator& generator,
                    void* data,
                    size_t size)
{
    if (cfg.shape.size() != 3)
    {
        WARNING("[Generator][DP][MatMul] Tensor shape expected 3 dimensions.");
        return false;
    }
    if (cfg.inputPos > 1 || cfg.inputPos < 0)
    {
        WARNING("[Generator][DP][MatMul] Invalid input tensor slot position to operator.");
        return false;
    }

    switch (cfg.dataType)
    {
        case DType::DType_FP32: {
            float* outData = reinterpret_cast<float*>(data);
            (cfg.inputPos == 0) ? generateMatMulA(cfg, generator, outData, size)
                                : generateMatMulB(cfg, generator, outData, size);
            break;
        }
        case DType::DType_FP16: {
            half_float::half* outData = reinterpret_cast<half_float::half*>(data);
            (cfg.inputPos == 0) ? generateMatMulA(cfg, generator, outData, size)
                                : generateMatMulB(cfg, generator, outData, size);
            break;
        }
        default:
            WARNING("[Generator][DP][MatMul] Only supports FP32 or FP16.");
            return false;
    }

    return true;
}
//---------------------------------------------------------------------------//
//                              Conv2D                                       //
//---------------------------------------------------------------------------//

template <typename DataType>
bool generateConv2DInput(const TosaReference::GenerateConfig& cfg,
                         TosaReference::IDotProductGenerator& generator,
                         DataType* data,
                         size_t size)
{
    if (cfg.dotProductInfo.kernel.size() != 2 || cfg.dotProductInfo.kernel[0] <= 0 || cfg.dotProductInfo.kernel[1] <= 0)
    {
        WARNING("[Generator][DP][Conv2D][Input] Missing or incorrect kernel size information.");
        return false;
    }
    if (cfg.shape.size() != 4)
    {
        WARNING("[Generator][DP][Conv2D][Input] Tensor shape expected 4 dimensions.");
        return false;
    }

    const int64_t T   = TosaReference::numElementsFromShape(cfg.shape);
    const uint32_t IH = cfg.shape[1];
    const uint32_t IW = cfg.shape[2];
    const uint32_t IC = cfg.shape[3];
    const uint32_t KH = cfg.dotProductInfo.kernel[0];
    const uint32_t KW = cfg.dotProductInfo.kernel[1];

    for (int64_t t = 0; t < T; ++t)
    {
        uint32_t ic = t % IC;
        uint32_t ix = (t / IC) % IW;
        uint32_t iy = ((t / IC) / IW) % IH;
        uint32_t k  = ((iy % KH) * KW + (ix % KW)) * IC + ic;

        data[t] = static_cast<DataType>(generator(k));
    }
    return true;
}

template <typename DataType>
bool generateConv2DWeight(const TosaReference::GenerateConfig& cfg,
                          TosaReference::IDotProductGenerator& generator,
                          DataType* data,
                          size_t size)
{
    if (cfg.shape.size() != 4)
    {
        WARNING("[Generator][DP][Conv2D][Weight] Tensor shape expected 4 dimensions.");
        return false;
    }

    const int64_t T   = TosaReference::numElementsFromShape(cfg.shape);
    const uint32_t KH = cfg.shape[1];
    const uint32_t KW = cfg.shape[2];
    const uint32_t IC = cfg.shape[3];

    for (int64_t t = 0; t < T; ++t)
    {
        uint32_t ic = t % IC;
        uint32_t kx = (t / IC) % KW;
        uint32_t ky = ((t / IC) / KW) % KH;
        uint32_t k  = (ky * KW + kx) * IC + ic;

        data[t] = static_cast<DataType>(generator(k));
    }
    return true;
}

template <typename DataType>
bool generateConv2DBias(const TosaReference::GenerateConfig& cfg,
                        TosaReference::IDotProductGenerator& generator,
                        DataType* data,
                        size_t size)
{
    if (cfg.shape.size() != 1)
    {
        WARNING("[Generator][DP][Conv2D][Bias] Tensor shape expected 1 dimension.");
        return false;
    }

    const uint32_t T = cfg.shape[0];

    for (uint32_t t = 0; t < T; ++t)
    {
        data[t] = static_cast<DataType>(generator(2));
    }
    return true;
}

bool generateConv2D(const TosaReference::GenerateConfig& cfg,
                    TosaReference::IDotProductGenerator& generator,
                    void* data,
                    size_t size)
{
    switch (cfg.dataType)
    {
        case DType::DType_FP32: {
            float* outData = reinterpret_cast<float*>(data);
            switch (cfg.inputPos)
            {
                case 0:
                    return generateConv2DInput(cfg, generator, outData, size);
                case 1:
                    return generateConv2DWeight(cfg, generator, outData, size);
                case 2:
                    return generateConv2DBias(cfg, generator, outData, size);
                default:
                    WARNING("[Generator][DP][Conv2D] Invalid input tensor slot position to operator.");
                    return false;
            }
            break;
        }
        case DType::DType_FP16: {
            half_float::half* outData = reinterpret_cast<half_float::half*>(data);
            switch (cfg.inputPos)
            {
                case 0:
                    return generateConv2DInput(cfg, generator, outData, size);
                case 1:
                    return generateConv2DWeight(cfg, generator, outData, size);
                case 2:
                    return generateConv2DBias(cfg, generator, outData, size);
                default:
                    WARNING("[Generator][DP][Conv2D] Invalid input tensor slot position to operator.");
                    return false;
            }
            break;
        }
        default:
            WARNING("[Generator][DP][Conv2D] Only supports FP32 or FP16.");
            return false;
    }
}
//---------------------------------------------------------------------------//
//                              Reduce Sum                                   //
//---------------------------------------------------------------------------//

template <typename DataType>
void generateReduceSumData(const TosaReference::GenerateConfig& cfg,
                           TosaReference::IDotProductGenerator& generator,
                           DataType* data,
                           size_t size)
{
    const int64_t T     = TosaReference::numElementsFromShape(cfg.shape);
    const uint32_t axis = cfg.dotProductInfo.axis;

    for (int64_t t = 0; t < T; ++t)
    {
        uint64_t k = t;
        for (uint32_t d = cfg.shape.size() - 1; d > axis; --d)
        {
            k = k / cfg.shape[d];
        }
        k = k % cfg.shape[axis];

        data[t] = static_cast<DataType>(generator(static_cast<int32_t>(k)));
    }
}

bool generateReduceSum(const TosaReference::GenerateConfig& cfg,
                       TosaReference::IDotProductGenerator& generator,
                       void* data,
                       size_t size)
{
    if (cfg.inputPos != 0)
    {
        WARNING("[Generator][DP][ReduceSum] Invalid input tensor slot position to operator.");
        return false;
    }
    if (cfg.dotProductInfo.axis < 0 || static_cast<size_t>(cfg.dotProductInfo.axis) >= cfg.shape.size())
    {
        WARNING("[Generator][DP][ReduceSum] Invalid axis %d.", cfg.dotProductInfo.axis);
        return false;
    }

    switch (cfg.dataType)
    {
        case DType::DType_FP32: {
            float* outData = reinterpret_cast<float*>(data);
            generateReduceSumData(cfg, generator, outData, size);
            break;
        }
        case DType::DType_FP16: {
            half_float::half* outData = reinterpret_cast<half_float::half*>(data);
            generateReduceSumData(cfg, generator, outData, size);
            break;
        }
        default:
            WARNING("[Generator][DP][ReduceSum] Only supports FP32 or FP16.");
            return false;
    }

    return true;
}
//---------------------------------------------------------------------------//
//                              Fully Connected                              //
//---------------------------------------------------------------------------//

template <typename DataType>
bool generateFullyConnectedInput(const TosaReference::GenerateConfig& cfg,
                                 TosaReference::IDotProductGenerator& generator,
                                 DataType* data,
                                 size_t size)
{
    if (cfg.shape.size() != 2)
    {
        WARNING("[Generator][DP][FullyConnected][Input] Tensor shape expected 2 dimensions.");
        return false;
    }

    const int64_t T   = TosaReference::numElementsFromShape(cfg.shape);
    const uint32_t IC = cfg.shape[1];

    for (int64_t t = 0; t < T; ++t)
    {
        uint32_t k = t % IC;

        data[t] = static_cast<DataType>(generator(k));
    }
    return true;
}

template <typename DataType>
bool generateFullyConnectedWeight(const TosaReference::GenerateConfig& cfg,
                                  TosaReference::IDotProductGenerator& generator,
                                  DataType* data,
                                  size_t size)
{
    if (cfg.shape.size() != 2)
    {
        WARNING("[Generator][DP][FullyConnected][Weight] Tensor shape expected 2 dimensions.");
        return false;
    }

    const int64_t T   = TosaReference::numElementsFromShape(cfg.shape);
    const uint32_t IC = cfg.shape[1];

    for (int64_t t = 0; t < T; ++t)
    {
        uint32_t k = t % IC;

        data[t] = static_cast<DataType>(generator(k));
    }
    return true;
}

template <typename DataType>
bool generateFullyConnectedBias(const TosaReference::GenerateConfig& cfg,
                                TosaReference::IDotProductGenerator& generator,
                                DataType* data,
                                size_t size)
{
    if (cfg.shape.size() != 1)
    {
        WARNING("[Generator][DP][FullyConnected][Bias] Tensor shape expected 1 dimension.");
        return false;
    }

    const uint32_t T = cfg.shape[0];

    for (uint32_t t = 0; t < T; ++t)
    {
        data[t] = static_cast<DataType>(generator(2));
    }
    return true;
}

bool generateFullyConnected(const TosaReference::GenerateConfig& cfg,
                            TosaReference::IDotProductGenerator& generator,
                            void* data,
                            size_t size)
{
    switch (cfg.dataType)
    {
        case DType::DType_FP32: {
            float* outData = reinterpret_cast<float*>(data);
            switch (cfg.inputPos)
            {
                case 0:
                    return generateFullyConnectedInput(cfg, generator, outData, size);
                case 1:
                    return generateFullyConnectedWeight(cfg, generator, outData, size);
                case 2:
                    return generateFullyConnectedBias(cfg, generator, outData, size);
                default:
                    WARNING("[Generator][DP][FullyConnected] Invalid input tensor slot position to operator.");
                    return false;
            }
            break;
        }
        case DType::DType_FP16: {
            half_float::half* outData = reinterpret_cast<half_float::half*>(data);
            switch (cfg.inputPos)
            {
                case 0:
                    return generateFullyConnectedInput(cfg, generator, outData, size);
                case 1:
                    return generateFullyConnectedWeight(cfg, generator, outData, size);
                case 2:
                    return generateFullyConnectedBias(cfg, generator, outData, size);
                default:
                    WARNING("[Generator][DP][FullyConnected] Invalid input tensor slot position to operator.");
                    return false;
            }
            break;
        }
        default:
            WARNING("[Generator][DP][FullyConnected] Only supports FP32 or FP16.");
            return false;
    }
}
//---------------------------------------------------------------------------//
//                              Avg Pool 2D                                  //
//---------------------------------------------------------------------------//

template <typename DataType>
void generateAvgPool2DData(const TosaReference::GenerateConfig& cfg,
                           TosaReference::IDotProductGenerator& generator,
                           DataType* data,
                           size_t size)
{
    const int64_t T   = TosaReference::numElementsFromShape(cfg.shape);
    const uint32_t IH = cfg.shape[1];
    const uint32_t IW = cfg.shape[2];
    const uint32_t C  = cfg.shape[3];
    const uint32_t KY = cfg.dotProductInfo.kernel[0];
    const uint32_t KX = cfg.dotProductInfo.kernel[1];

    for (int64_t t = 0; t < T; ++t)
    {
        uint32_t c  = t % C;
        uint32_t ix = (t / C) % IW;
        uint32_t iy = ((t / C) / IW) % IH;
        uint32_t k  = ((iy % KY) * KX + (ix % KX)) * C + c;

        data[t] = static_cast<DataType>(generator(k));
    }
}

bool generateAvgPool2D(const TosaReference::GenerateConfig& cfg,
                       TosaReference::IDotProductGenerator& generator,
                       void* data,
                       size_t size)
{
    if (cfg.inputPos != 0)
    {
        WARNING("[Generator][DP][AvgPool2D] Invalid input tensor slot position to operator.");
        return false;
    }
    if (cfg.dotProductInfo.kernel.size() != 2 || cfg.dotProductInfo.kernel[0] <= 0 || cfg.dotProductInfo.kernel[1] <= 0)
    {
        WARNING("[Generator][DP][AvgPool2D] Missing or incorrect kernel size information.");
        return false;
    }
    if (cfg.shape.size() != 4)
    {
        WARNING("[Generator][DP][AvgPool2D] Tensor shape expected 4 dimensions.");
        return false;
    }

    switch (cfg.dataType)
    {
        case DType::DType_FP32: {
            float* outData = reinterpret_cast<float*>(data);
            generateAvgPool2DData(cfg, generator, outData, size);
            break;
        }
        case DType::DType_FP16: {
            half_float::half* outData = reinterpret_cast<half_float::half*>(data);
            generateAvgPool2DData(cfg, generator, outData, size);
            break;
        }
        default:
            WARNING("[Generator][DP][AvgPool2D] Only supports FP32 or FP16.");
            return false;
    }

    return true;
}
//---------------------------------------------------------------------------//
//                              Depthwise Conv2D                             //
//---------------------------------------------------------------------------//

template <typename DataType>
bool generateDepthwiseConv2DInput(const TosaReference::GenerateConfig& cfg,
                                  TosaReference::IDotProductGenerator& generator,
                                  DataType* data,
                                  size_t size)
{
    if (cfg.dotProductInfo.kernel.size() != 2 || cfg.dotProductInfo.kernel[0] <= 0 || cfg.dotProductInfo.kernel[1] <= 0)
    {
        WARNING("[Generator][DP][DWConv2D][Input] Missing or incorrect kernel size information.");
        return false;
    }
    if (cfg.shape.size() != 4)
    {
        WARNING("[Generator][DP][DWConv2D][Input] Tensor shape expected 4 dimensions.");
        return false;
    }

    const int64_t T   = TosaReference::numElementsFromShape(cfg.shape);
    const uint32_t IH = cfg.shape[1];
    const uint32_t IW = cfg.shape[2];
    const uint32_t C  = cfg.shape[3];
    const uint32_t KH = cfg.dotProductInfo.kernel[0];
    const uint32_t KW = cfg.dotProductInfo.kernel[1];

    for (int64_t t = 0; t < T; ++t)
    {
        uint32_t ix = (t / C) % IW;
        uint32_t iy = ((t / C) / IW) % IH;
        uint32_t k  = ((iy % KH) * KW + (ix % KW));

        data[t] = static_cast<DataType>(generator(k));
    }
    return true;
}

template <typename DataType>
bool generateDepthwiseConv2DWeight(const TosaReference::GenerateConfig& cfg,
                                   TosaReference::IDotProductGenerator& generator,
                                   DataType* data,
                                   size_t size)
{
    if (cfg.shape.size() != 4)
    {
        WARNING("[Generator][DP][DWConv2D][Weight] Tensor shape expected 4 dimensions.");
        return false;
    }

    const int64_t T   = TosaReference::numElementsFromShape(cfg.shape);
    const uint32_t KH = cfg.shape[0];
    const uint32_t KW = cfg.shape[1];
    const uint32_t C  = cfg.shape[2];
    const uint32_t M  = cfg.shape[3];

    for (int64_t t = 0; t < T; ++t)
    {
        uint32_t kx = ((t / M) / C) % KW;
        uint32_t ky = (((t / M) / C) / KW) % KH;
        uint32_t k  = (ky * KW + kx);

        data[t] = static_cast<DataType>(generator(k));
    }
    return true;
}

template <typename DataType>
bool generateDepthwiseConv2DBias(const TosaReference::GenerateConfig& cfg,
                                 TosaReference::IDotProductGenerator& generator,
                                 DataType* data,
                                 size_t size)
{
    if (cfg.shape.size() != 1)
    {
        WARNING("[Generator][DP][DWConv2D][Bias] Tensor shape expected 1 dimension.");
        return false;
    }

    const uint32_t T = cfg.shape[0];

    for (uint32_t t = 0; t < T; ++t)
    {
        data[t] = static_cast<DataType>(generator(2));
    }
    return true;
}

bool generateDepthwiseConv2D(const TosaReference::GenerateConfig& cfg,
                             TosaReference::IDotProductGenerator& generator,
                             void* data,
                             size_t size)
{
    switch (cfg.dataType)
    {
        case DType::DType_FP32: {
            float* outData = reinterpret_cast<float*>(data);
            switch (cfg.inputPos)
            {
                case 0:
                    return generateDepthwiseConv2DInput(cfg, generator, outData, size);
                case 1:
                    return generateDepthwiseConv2DWeight(cfg, generator, outData, size);
                case 2:
                    return generateDepthwiseConv2DBias(cfg, generator, outData, size);
                default:
                    WARNING("[Generator][DP][DWConv2D] Invalid input tensor slot position to operator.");
                    return false;
            }
            break;
        }
        case DType::DType_FP16: {
            half_float::half* outData = reinterpret_cast<half_float::half*>(data);
            switch (cfg.inputPos)
            {
                case 0:
                    return generateDepthwiseConv2DInput(cfg, generator, outData, size);
                case 1:
                    return generateDepthwiseConv2DWeight(cfg, generator, outData, size);
                case 2:
                    return generateDepthwiseConv2DBias(cfg, generator, outData, size);
                default:
                    WARNING("[Generator][DP][DWConv2D] Invalid input tensor slot position to operator.");
                    return false;
            }
            break;
        }
        default:
            WARNING("[Generator][DP][DWConv2D] Only supports FP32 or FP16.");
            return false;
    }
}
//---------------------------------------------------------------------------//
//                              Transpose Conv2D                             //
//---------------------------------------------------------------------------//

template <typename DataType>
bool generateTransposeConv2DInput(const TosaReference::GenerateConfig& cfg,
                                  TosaReference::IDotProductGenerator& generator,
                                  DataType* data,
                                  size_t size)
{
    if (cfg.dotProductInfo.kernel.size() != 2 || cfg.dotProductInfo.kernel[0] <= 0 || cfg.dotProductInfo.kernel[1] <= 0)
    {
        WARNING("[Generator][DP][TConv2D][Input] Missing or incorrect kernel size information.");
        return false;
    }
    if (cfg.shape.size() != 4)
    {
        WARNING("[Generator][DP][TConv2D][Input] Tensor shape expected 4 dimensions.");
        return false;
    }

    const int64_t T   = TosaReference::numElementsFromShape(cfg.shape);
    const uint32_t IH = cfg.shape[1];
    const uint32_t IW = cfg.shape[2];
    const uint32_t IC = cfg.shape[3];
    const uint32_t KH = cfg.dotProductInfo.kernel[0];
    const uint32_t KW = cfg.dotProductInfo.kernel[1];

    for (int64_t t = 0; t < T; ++t)
    {
        uint32_t ic = t % IC;
        uint32_t ix = (t / IC) % IW;
        uint32_t iy = ((t / IC) / IW) % IH;
        uint32_t k  = ((iy % KH) * KW + (ix % KW)) * IC + ic;

        data[t] = static_cast<DataType>(generator(k));
    }
    return true;
}

template <typename DataType>
bool generateTransposeConv2DWeight(const TosaReference::GenerateConfig& cfg,
                                   TosaReference::IDotProductGenerator& generator,
                                   DataType* data,
                                   size_t size)
{
    if (cfg.shape.size() != 4)
    {
        WARNING("[Generator][DP][TConv2D][Weight] Tensor shape expected 4 dimensions.");
        return false;
    }

    const int64_t T   = TosaReference::numElementsFromShape(cfg.shape);
    const uint32_t KH = cfg.shape[1];
    const uint32_t KW = cfg.shape[2];
    const uint32_t IC = cfg.shape[3];

    for (int64_t t = 0; t < T; ++t)
    {
        uint32_t ic = t % IC;
        uint32_t kx = (t / IC) % KW;
        uint32_t ky = ((t / IC) / KW) % KH;
        uint32_t k  = (ky * KW + kx) * IC + ic;

        data[t] = static_cast<DataType>(generator(k));
    }
    return true;
}

template <typename DataType>
bool generateTransposeConv2DBias(const TosaReference::GenerateConfig& cfg,
                                 TosaReference::IDotProductGenerator& generator,
                                 DataType* data,
                                 size_t size)
{
    if (cfg.shape.size() != 1)
    {
        WARNING("[Generator][DP][TConv2D][Bias] Tensor shape expected 1 dimension.");
        return false;
    }

    const uint32_t T = cfg.shape[0];

    for (uint32_t t = 0; t < T; ++t)
    {
        data[t] = static_cast<DataType>(generator(2));
    }
    return true;
}

bool generateTransposeConv2D(const TosaReference::GenerateConfig& cfg,
                             TosaReference::IDotProductGenerator& generator,
                             void* data,
                             size_t size)
{
    switch (cfg.dataType)
    {
        case DType::DType_FP32: {
            float* outData = reinterpret_cast<float*>(data);
            switch (cfg.inputPos)
            {
                case 0:
                    return generateTransposeConv2DInput(cfg, generator, outData, size);
                case 1:
                    return generateTransposeConv2DWeight(cfg, generator, outData, size);
                case 2:
                    return generateTransposeConv2DBias(cfg, generator, outData, size);
                default:
                    WARNING("[Generator][DP][TConv2D] Invalid input tensor slot position to operator.");
                    return false;
            }
            break;
        }
        case DType::DType_FP16: {
            half_float::half* outData = reinterpret_cast<half_float::half*>(data);
            switch (cfg.inputPos)
            {
                case 0:
                    return generateTransposeConv2DInput(cfg, generator, outData, size);
                case 1:
                    return generateTransposeConv2DWeight(cfg, generator, outData, size);
                case 2:
                    return generateTransposeConv2DBias(cfg, generator, outData, size);
                default:
                    WARNING("[Generator][DP][TConv2D] Invalid input tensor slot position to operator.");
                    return false;
            }
            break;
        }
        default:
            WARNING("[Generator][DP][TConv2D] Only supports FP32 or FP16.");
            return false;
    }
}
//---------------------------------------------------------------------------//
//                              FFT2D                                        //
//---------------------------------------------------------------------------//

template <typename DataType>
bool generateFFT2DReal(const TosaReference::GenerateConfig& cfg,
                       TosaReference::IDotProductGenerator& generator,
                       DataType* data,
                       size_t size)
{
    const int64_t T  = TosaReference::numElementsFromShape(cfg.shape);
    const uint32_t H = cfg.shape[1];
    const uint32_t W = cfg.shape[2];

    for (int64_t t = 0; t < T; ++t)
    {
        uint32_t x = t % W;
        uint32_t y = (t / W) % H;
        uint32_t k = y * W + x;

        data[t] = static_cast<DataType>(generator(k));
    }
    return true;
}

template <typename DataType>
bool generateFFT2DImag(const TosaReference::GenerateConfig& cfg,
                       TosaReference::IDotProductGenerator& generator,
                       DataType* data,
                       size_t size)
{
    const int64_t T  = TosaReference::numElementsFromShape(cfg.shape);
    const uint32_t H = cfg.shape[1];
    const uint32_t W = cfg.shape[2];

    // The index expression of ((1*N+n)*H+y)*W+x in the spec equates to
    // using the values after those used for the Real tensor, but we need
    // to iterate through all those values to get to the Imaginary data
    for (int64_t n = 0; n < 2; ++n)
    {
        for (int64_t t = 0; t < T; ++t)
        {
            uint32_t x = t % W;
            uint32_t y = (t / W) % H;
            uint32_t k = y * W + x;

            data[t] = static_cast<DataType>(generator(k));
        }
    }
    return true;
}

bool generateFFT2D(const TosaReference::GenerateConfig& cfg,
                   TosaReference::IDotProductGenerator& generator,
                   void* data,
                   size_t size)
{
    if (cfg.shape.size() != 3)
    {
        WARNING("[Generator][DP][FFT2D] Tensor shape expected 3 dimensions.");
        return false;
    }

    switch (cfg.dataType)
    {
        case DType::DType_FP32: {
            float* outData = reinterpret_cast<float*>(data);
            switch (cfg.inputPos)
            {
                case 0:
                    return generateFFT2DReal(cfg, generator, outData, size);
                case 1:
                    return generateFFT2DImag(cfg, generator, outData, size);
                default:
                    WARNING("[Generator][DP][FFT2D] Invalid input tensor slot position to operator.");
                    return false;
            }
            break;
        }
        default:
            WARNING("[Generator][DP][FFT2D] Only supports FP32.");
            return false;
    }

    return true;
}
}    // namespace

namespace TosaReference
{

bool generateDotProduct(const GenerateConfig& cfg, void* data, size_t size)
{
    auto generator = pickDotProductGenerator(cfg);
    if (!generator)
    {
        WARNING("[Generator][DP] Requested generator could not be created!");
        return false;
    }
    if (cfg.dotProductInfo.ks <= 0)
    {
        WARNING("[Generator][DP] Invalid test set kernel size %d.", cfg.dotProductInfo.ks);
        return false;
    }

    // Select which generator to use
    switch (cfg.opType)
    {
        case tosa::Op_MATMUL:
            return generateMatMul(cfg, *generator, data, size);
        case tosa::Op_CONV2D:
            return generateConv2D(cfg, *generator, data, size);
        case tosa::Op_REDUCE_SUM:
            return generateReduceSum(cfg, *generator, data, size);
        case tosa::Op_FULLY_CONNECTED:
            return generateFullyConnected(cfg, *generator, data, size);
        case tosa::Op_AVG_POOL2D:
            return generateAvgPool2D(cfg, *generator, data, size);
        case tosa::Op_DEPTHWISE_CONV2D:
            return generateDepthwiseConv2D(cfg, *generator, data, size);
        case tosa::Op_TRANSPOSE_CONV2D:
            return generateTransposeConv2D(cfg, *generator, data, size);
        case tosa::Op_FFT2D:
            return generateFFT2D(cfg, *generator, data, size);
        default:
            WARNING("[Generator][DP] Unsupported operator.");
            return false;
    }

    return false;
}
}    // namespace TosaReference
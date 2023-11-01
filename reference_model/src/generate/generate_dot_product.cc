// Copyright (c) 2023, ARM Limited.
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

namespace
{
//---------------------------------------------------------------------------//
//                              MatMul                                       //
//---------------------------------------------------------------------------//

void generateMatMulA(const TosaReference::GenerateConfig& cfg,
                     TosaReference::IDotProductGenerator& generator,
                     void* data,
                     size_t size)
{
    float* a         = reinterpret_cast<float*>(data);
    const uint32_t T = cfg.shape[0] * cfg.shape[1] * cfg.shape[2];
    const uint32_t C = cfg.shape[2];

    for (uint32_t t = 0; t < T; ++t)
    {
        a[t] = generator(t % C);    // k = c
    }
}

void generateMatMulB(const TosaReference::GenerateConfig& cfg,
                     TosaReference::IDotProductGenerator& generator,
                     void* data,
                     size_t size)
{
    float* b         = reinterpret_cast<float*>(data);
    const uint32_t T = cfg.shape[0] * cfg.shape[1] * cfg.shape[2];
    const uint32_t C = cfg.shape[1];
    const uint32_t W = cfg.shape[2];

    for (uint32_t t = 0; t < T; ++t)
    {
        b[t] = generator((t / W) % C);    // k = c
    }
}

bool generateMatMul(const TosaReference::GenerateConfig& cfg,
                    TosaReference::IDotProductGenerator& generator,
                    void* data,
                    size_t size)
{
    if (cfg.dataType != DType::DType_FP32)
    {
        WARNING("[Generator][DP][MatMul] Only supports FP32.");
        return false;
    }
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

    (cfg.inputPos == 0) ? generateMatMulA(cfg, generator, data, size) : generateMatMulB(cfg, generator, data, size);

    return true;
}
//---------------------------------------------------------------------------//
//                              Conv2D                                       //
//---------------------------------------------------------------------------//

bool generateConv2DInput(const TosaReference::GenerateConfig& cfg,
                         TosaReference::IDotProductGenerator& generator,
                         void* data,
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

    float* input      = reinterpret_cast<float*>(data);
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

        input[t] = generator(k);
    }
    return true;
}

bool generateConv2DWeight(const TosaReference::GenerateConfig& cfg,
                          TosaReference::IDotProductGenerator& generator,
                          void* data,
                          size_t size)
{
    if (cfg.shape.size() != 4)
    {
        WARNING("[Generator][DP][Conv2D][Weight] Tensor shape expected 4 dimensions.");
        return false;
    }

    float* weight     = reinterpret_cast<float*>(data);
    const int64_t T   = TosaReference::numElementsFromShape(cfg.shape);
    const uint32_t KH = cfg.shape[1];
    const uint32_t KW = cfg.shape[2];
    const uint32_t IC = cfg.shape[3];

    for (int64_t t = 0; t < T; ++t)
    {
        uint32_t ic = t % IC;
        uint32_t kx = (t / IC) % KW;
        uint32_t ky = ((t / IC) / KW) % KH;
        uint32_t k  = (ky + KW * kx) * IC + ic;

        weight[t] = generator(k);
    }
    return true;
}

bool generateConv2DBias(const TosaReference::GenerateConfig& cfg,
                        TosaReference::IDotProductGenerator& generator,
                        void* data,
                        size_t size)
{
    if (cfg.shape.size() != 1)
    {
        WARNING("[Generator][DP][Conv2D][Bias] Tensor shape expected 1 dimension.");
        return false;
    }

    float* bias      = reinterpret_cast<float*>(data);
    const uint32_t T = cfg.shape[0];

    for (uint32_t t = 0; t < T; ++t)
    {
        bias[t] = generator(2);
    }
    return true;
}

bool generateConv2D(const TosaReference::GenerateConfig& cfg,
                    TosaReference::IDotProductGenerator& generator,
                    void* data,
                    size_t size)
{
    if (cfg.dataType != DType::DType_FP32)
    {
        WARNING("[Generator][DP][Conv2D] Only supports FP32.");
        return false;
    }
    switch (cfg.inputPos)
    {
        case 0:
            return generateConv2DInput(cfg, generator, data, size);
        case 1:
            return generateConv2DWeight(cfg, generator, data, size);
        case 2:
            return generateConv2DBias(cfg, generator, data, size);
        default:
            WARNING("[Generator][DP][Conv2D] Invalid input tensor slot position to operator.");
            return false;
    }
}
//---------------------------------------------------------------------------//
//                              Reduce Sum                                   //
//---------------------------------------------------------------------------//

bool generateReduceSum(const TosaReference::GenerateConfig& cfg,
                       TosaReference::IDotProductGenerator& generator,
                       void* data,
                       size_t size)
{
    if (cfg.dataType != DType::DType_FP32)
    {
        WARNING("[Generator][DP][ReduceSum] Only supports FP32.");
        return false;
    }
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

    float* input        = reinterpret_cast<float*>(data);
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

        input[t] = generator(static_cast<int32_t>(k));
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
        return 0;
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
        default:
            WARNING("[Generator][DP] Unsupported operator.");
            return false;
    }

    return false;
}
}    // namespace TosaReference
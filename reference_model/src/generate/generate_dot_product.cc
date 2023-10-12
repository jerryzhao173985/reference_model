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
        default:
            WARNING("[Generator][DP] Unsupported operator.");
            return false;
    }

    return false;
}
}    // namespace TosaReference
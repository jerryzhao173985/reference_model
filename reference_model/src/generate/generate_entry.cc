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

#include "generate.h"

#include "generate_dot_product.h"
#include "generate_fixed_data.h"
#include "generate_pseudo_random.h"
#include "generate_utils.h"

#include "func_debug.h"
#include "model_common.h"

namespace TosaReference
{

bool generate(const GenerateConfig& cfg, void* data, size_t size)
{
    switch (cfg.generatorType)
    {
        case GeneratorType::DotProduct: {
            return generateDotProduct(cfg, data, size);
            break;
        }
        case GeneratorType::PseudoRandom: {
            return generatePseudoRandom(cfg, data, size);
            break;
        }
        case GeneratorType::FixedData: {
            return generateFixedData(cfg, data, size);
            break;
        }
        default: {
            WARNING("[Generator] Unsupported generation mode.");
            break;
        }
    }
    return false;
}

}    // namespace TosaReference

extern "C"
{
    bool tgd_generate_data(const char* config_json, const char* tensor_name, void* data, size_t size)
    {
        // Check inputs for nullptr
        if (!config_json || !tensor_name || !data)
        {
            WARNING("[Generator] One of the inputs is missing.");
            return false;
        }

        // Check JSON config validity
        auto cfg = TosaReference::parseGenerateConfig(config_json, tensor_name);
        if (!cfg)
        {
            return false;
        }

        // Check size
        const size_t totalBytesNeeded =
            TosaReference::numElementsFromShape(cfg->shape) * TosaReference::elementSizeFromType(cfg->dataType);
        if (totalBytesNeeded > size)
        {
            WARNING("[Generator] Not enough space in provided buffer.");
            return false;
        }

        // Run generator
        return generate(cfg.value(), data, size);
    }
}    // extern "C"

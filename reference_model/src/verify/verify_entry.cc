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

#include "verify.h"

#include "func_debug.h"
#include "model_common.h"
#include "verifiers.h"
#include "verify_utils.h"

#include <vector>

namespace TosaReference
{

bool verify(const CTensor* ref, const CTensor* refBnd, const CTensor* imp, const VerifyConfig& cfg)
{
    switch (cfg.mode)
    {
        case VerifyMode::DotProduct: {
            return verifyDotProduct(ref, refBnd, imp, cfg.dotProductInfo);
            break;
        }
        default: {
            WARNING("unsupported verification mode.");
            break;
        }
    }
    return false;
}

}    // namespace TosaReference

extern "C"
{
    bool tvf_verify_data(const tosa_tensor_t* ref,
                         const tosa_tensor_t* ref_bnd,
                         const tosa_tensor_t* imp,
                         const char* config_json)
    {
        // Check inputs for nullptr
        if (!ref || !imp || !config_json)
        {
            WARNING("one of the inputs is missing.");
            return false;
        }

        // Extract verification config
        if (!ref->name)
        {
            WARNING("tensor name is not specified.");
            return false;
        }
        auto cfg = TosaReference::parseVerifyConfig(ref->name, config_json);
        if (!cfg)
        {
            WARNING("invalid json config.");
            return false;
        }

        // Validate shape
        if (ref->num_dims != imp->num_dims)
        {
            WARNING("tensors have different number of dimensions.");
            return false;
        }
        if (!ref->shape || !imp->shape)
        {
            WARNING("one of tensors' shape is missing.");
            return false;
        }
        if (std::vector(ref->shape, ref->shape + ref->num_dims) != std::vector(imp->shape, imp->shape + imp->num_dims))
        {
            WARNING("tensors have different shapes.");
            return false;
        }

        // Run verification
        return verify(ref, ref_bnd, imp, *cfg);
    }
}    // extern "C"

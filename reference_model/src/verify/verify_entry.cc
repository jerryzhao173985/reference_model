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
        }
        case VerifyMode::Exact: {
            return verifyExact(ref, imp);
        }
        case VerifyMode::ReduceProduct: {
            return verifyReduceProduct(ref, imp, cfg.reduceProductInfo);
        }
        case VerifyMode::Ulp: {
            return verifyULP(ref, imp, cfg.ulpInfo);
        }
        case VerifyMode::AbsError: {
            return verifyAbsError(ref, refBnd, imp, cfg.absErrorInfo);
        }
        case VerifyMode::Relative: {
            return verifyRelative(ref, imp, cfg.relativeInfo);
        }
        case VerifyMode::FpSpecial: {
            return verifyFpSpecial(ref, refBnd, imp);
        }
        default: {
            WARNING("[Verifier] Unsupported verification mode.");
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
            WARNING("[Verifier] One of the inputs is missing.");
            return false;
        }

        // Extract verification config
        if (!ref->name)
        {
            WARNING("[Verifier] Tensor name is not specified.");
            return false;
        }
        auto cfg = TosaReference::parseVerifyConfig(ref->name, config_json);
        if (!cfg)
        {
            return false;
        }

        // Validate shape
        if (ref->num_dims != imp->num_dims)
        {
            WARNING("[Verifier] Tensors have different number of dimensions.");
            return false;
        }
        //checks shape info is given, but still allows rank 0 tensors to pass
        if ((!ref->shape || !imp->shape) && ref->num_dims != 0)
        {
            WARNING("[Verifier] One of tensors' shape is missing.");
            return false;
        }
        if (std::vector(ref->shape, ref->shape + ref->num_dims) != std::vector(imp->shape, imp->shape + imp->num_dims))
        {
            WARNING("[Verifier] Tensors have different shapes.");
            return false;
        }
        // Validate data-type

        tosa_datatype_t imp_type = imp->data_type;
        //checks if all datatypes are supported, check verif_utils.cc to add a datatype
        if (cfg->dataType == DType::DType_UNKNOWN)
        {
            WARNING("Unsupported data type in compliance configuration");
            return false;
        }

        if (cfg->dataType != TosaReference::mapToDType(imp_type))
        {
            WARNING("[Verifier] Incorrect implementation tensor data type.");
            return false;
        }
        if (((imp_type == tosa_datatype_fp16_t) || (imp_type == tosa_datatype_fp32_t) ||
             (imp_type == tosa_datatype_fp64_t) || (imp_type == tosa_datatype_fp8e4m3_t) ||
             (imp_type == tosa_datatype_fp8e5m2_t) || (imp_type == tosa_datatype_bf16_t)))
        {
            if ((ref->data_type != tosa_datatype_fp64_t))
            {
                WARNING("[Verifier] Reference tensor data type is not FP64, please use ref-model --precise_mode.");
                return false;
            }
        }
        else
        {
            if ((ref->data_type != imp_type))
            {
                WARNING("[Verifier] Reference and implementation data types do not match");
                return false;
            }
        }

        // Run verification
        return verify(ref, ref_bnd, imp, *cfg);
    }
}    // extern "C"

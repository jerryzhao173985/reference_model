// Copyright (c) 2023,2025, ARM Limited.
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
//===----------------------------------------------------------------------===//
//
// Verification functionality as per TOSA Specification
// Output Verification : Section 1.8.2
//
//===----------------------------------------------------------------------===//
#ifndef VERIFY_H
#define VERIFY_H

#include "config.h"
#include "types.h"
#include <cstdlib>

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */

    enum tvf_status_t : int
    {
        TVF_COMPLIANT     = EXIT_SUCCESS,
        TVF_ERROR         = EXIT_FAILURE,
        TVF_NON_COMPLIANT = 2,
    };

    /// \brief Perform compliance validation between a reference and a target output
    ///
    /// A compliance configuration is expected as it provides information about
    /// the type of validation to be performed alongside with all the relevant
    /// meta-data. Configuration is provided in JSON format.
    ///
    /// \param ref         Reference tensor to compare against
    /// \param ref_bnd     (Optional) Reference tensor when run on absolute inputs
    /// \param imp         Implementation resulting tensor
    /// \param config_json Compliance configuration that indicates how and what compliance need to be performed
    ///
    /// \return True in case of successful validation else false
    TOSA_EXPORT bool tvf_verify_data(const tosa_tensor_t* ref,
                                     const tosa_tensor_t* ref_bnd,
                                     const tosa_tensor_t* imp,
                                     const char* config_json);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif    // VERIFY_H

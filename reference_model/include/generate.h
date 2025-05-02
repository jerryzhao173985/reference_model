// Copyright (c) 2023, 2025, ARM Limited.
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
// Data generation functionality as per TOSA Specification (5.2)
//
//===----------------------------------------------------------------------===//
#ifndef GENERATE_H
#define GENERATE_H

#include "config.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */

    /// \brief Perform input data generation for a given tensor
    ///
    /// A configuration provides context about the type of generator to be used e.g. Pseudo-random
    /// alongside with information on the operator and the slot that the tensor is consumed by.
    ///
    /// \param config_json JSON configuration of the tensor that we need to generate data for
    /// \param tensor_name Name of the tensor to extract generator information
    /// \param data User-provided buffer to store the data to
    /// \param size Size of the provided buffer in bytes
    /// \return
    TOSA_EXPORT bool tgd_generate_data(const char* config_json, const char* tensor_name, void* data, size_t size);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif    // GENERATE_H

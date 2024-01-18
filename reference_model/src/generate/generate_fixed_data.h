// Copyright (c) 2024, ARM Limited.
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

#ifndef GENERATE_FIXED_DATA_H_
#define GENERATE_FIXED_DATA_H_

#include "generate_utils.h"

namespace TosaReference
{

/// \brief Perform fixed data generation
///
/// \param cfg Generator related meta-data
/// \param data Buffer to generate the data to
/// \param size Size of the buffer
///
/// \return True on successful generation
bool generateFixedData(const GenerateConfig& cfg, void* data, size_t size);

};    // namespace TosaReference

#endif    // GENERATE_FIXED_DATA_H_

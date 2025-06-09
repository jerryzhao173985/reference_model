
// Copyright (c) 2022, ARM Limited.
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

#ifndef GENERAL_UTILS_H_
#define GENERAL_UTILS_H_

#include "func_debug.h"

#include "numpy_utils.h"

namespace TosaReference
{

const uint32_t getElementCount(std::vector<int32_t>& shape)
{
    uint32_t elements = 1;
    for (size_t i = 0; i < shape.size(); i++)
    {
        elements *= static_cast<uint32_t>(shape[i]);
    }

    return elements;
}

template <typename T>
std::vector<T> readFromNpyFile(const char* filename, std::vector<int32_t>& shape)
{
    uint32_t elements = getElementCount(shape);
    std::vector<T> data(elements, 0);

    NumpyUtilities::NPError nperror = NumpyUtilities::readFromNpyFile(filename, elements, data.data());

    switch (nperror)
    {
        case NumpyUtilities::NO_ERROR:
            break;
        case NumpyUtilities::FILE_NOT_FOUND:
            FATAL_ERROR("readFromNpyFile: Cannot open file %s", filename);
        case NumpyUtilities::FILE_IO_ERROR:
            FATAL_ERROR("readFromNpyFile: IO error reading file: %s", filename);
        case NumpyUtilities::FILE_TYPE_MISMATCH:
            FATAL_ERROR("readFromNpyFile: Tensor type and Numpy file type mismatch for filename %s", filename);
        case NumpyUtilities::HEADER_PARSE_ERROR:
            FATAL_ERROR("Numpy header parsing error for file: %s", filename);
        case NumpyUtilities::BUFFER_SIZE_MISMATCH:
            FATAL_ERROR("Buffer size does not match numpy file size for filename %s", filename);
        default:
            FATAL_ERROR("Unknown error parsing Numpy file: %s", filename);
    }

    return data;
}

};    // namespace TosaReference

#endif

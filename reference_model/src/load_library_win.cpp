
// Copyright (c) 2023, 2025, Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "load_library.h"
#include <string>

LIBTYPE load_library_w(const char* libname)
{
    size_t outSize;
    auto const size{ std::string_view{ libname }.size() + 1 };
    wchar_t* l_libname = (wchar_t*)(sizeof(wchar_t) * size);

    mbstowcs_s(&outSize, l_libname, size, libname, size - 1);

    auto lib = LoadLibraryW(l_libname);

    free(l_libname);
    return lib;
}


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

HMODULE load_library_w(const char* libname)
{
    HMODULE handle = NULL;
    int sizeNeeded = MultiByteToWideChar(CP_UTF8, 0, libname, -1, nullptr, 0);
    if (sizeNeeded == 0)
    {
        return handle;
    }

    std::wstring wideStr(sizeNeeded, 0);
    MultiByteToWideChar(CP_UTF8, 0, libname, -1, &wideStr[0], sizeNeeded);

    handle = LoadLibraryW(wideStr.c_str());

    return handle;
}

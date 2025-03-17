
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

#ifndef LOAD_LIBRARY_H
#define LOAD_LIBRARY_H

#if defined(__linux__) || defined(__APPLE__)
#include <dlfcn.h>
#define LIBTYPE void*
#define OPENLIB(libname) dlopen((libname), RTLD_LAZY)
#define LIBFUNC(lib, fn) dlsym((lib), (fn))
#define CLOSELIB(lib) dlclose((lib))
#elif _WIN32
#define NOMINMAX
#include <windows.h>
#define LIBTYPE HINSTANCE
#define OPENLIB(libname) load_library_w(libname)
#define LIBFUNC(lib, fn) GetProcAddress((lib), (fn))
#define CLOSELIB(lib) FreeLibrary((lib))
LIBTYPE load_library_w(const char* libname);
#endif

#endif /* LOAD_LIBRARY_H */

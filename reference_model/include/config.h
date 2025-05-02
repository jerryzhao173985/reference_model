// Copyright (c) 2025, ARM Limited.
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

#ifndef TOSA_REFERENCE_CONFIG_H
#define TOSA_REFERENCE_CONFIG_H

// Eigen configuration options
#ifndef EIGEN_CORE_H
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int64_t
#else
#error "config.h contains Eigen configuration options, so should be included before Eigen related includes"
#endif

// DLL export
#ifdef _MSC_VER
#define TOSA_EXPORT __declspec(dllexport)
#else
#define TOSA_EXPORT
#endif

#endif    // TOSA_REFERENCE_CONFIG_H


// Copyright (c) 2022-2023, ARM Limited.
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

#ifndef TYPES_H_
#define TYPES_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */

    // Note status needs to be aligned with graph_status
    enum tosa_status_t
    {
        tosa_status_valid         = 0,
        tosa_status_unpredictable = 1,
        tosa_status_error         = 2
    };

    enum tosa_mode_t
    {
        tosa_mode_unknown  = 0,
        tosa_mode_nearest  = 1,
        tosa_mode_bilinear = 2,
        tosa_mode_min      = 3,
        tosa_mode_max      = 4
    };

    enum tosa_datatype_t
    {
        tosa_datatype_bf16_t   = 0,
        tosa_datatype_bool_t   = 1,
        tosa_datatype_fp16_t   = 2,
        tosa_datatype_fp32_t   = 3,
        tosa_datatype_int16_t  = 4,
        tosa_datatype_int32_t  = 5,
        tosa_datatype_int48_t  = 6,
        tosa_datatype_int4_t   = 7,
        tosa_datatype_int8_t   = 8,
        tosa_datatype_uint16_t = 9,
        tosa_datatype_uint8_t  = 10,
        tosa_datatype_shape_t  = 11,
        tosa_datatype_fp64_t   = 99
    };

    struct tosa_tensor_t
    {
        const char* name;
        int32_t* shape;
        int32_t num_dims;
        tosa_datatype_t data_type;
        uint8_t* data;
        size_t size;
    };

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif    // TYPES_H_
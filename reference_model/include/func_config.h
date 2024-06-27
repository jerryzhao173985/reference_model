
// Copyright (c) 2020-2023, ARM Limited.
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

#ifndef FUNC_CONFIG_H_
#define FUNC_CONFIG_H_

#include <iostream>
#include <stdio.h>

struct tosa_level_t
{
    int32_t MAX_RANK   = 0;
    int32_t MAX_KERNEL = 0;
    int32_t MAX_STRIDE = 0;
    int32_t MAX_SCALE  = 0;

    bool operator!=(const tosa_level_t& rhs)
    {
        return !(MAX_RANK == rhs.MAX_RANK && MAX_KERNEL == rhs.MAX_KERNEL && MAX_STRIDE == rhs.MAX_STRIDE &&
                 MAX_SCALE == rhs.MAX_SCALE);
    }
};

struct func_config_t
{
    std::string operator_fbs   = "tosa.fbs";
    std::string test_desc      = "desc.json";
    std::string flatbuffer_dir = "";
    std::string output_dir     = "";
    std::string tosa_file      = "";
    std::string ifm_name       = "";
    std::string ifm_file       = "";
    std::string ofm_name       = "";
    std::string ofm_file       = "";
    std::string variable_name  = "";
    std::string variable_file  = "";

    uint32_t eval                                  = 1;
    uint32_t validate_only                         = 0;
    uint32_t output_tensors                        = 1;
    uint32_t dump_intermediates                    = 0;
    uint32_t initialize_variable_tensor_from_numpy = 0;
    std::string fp_format                          = "0.5";
    std::string custom_op_lib_path                 = "";
    uint32_t precise_mode                          = 0;
    bool abs_mode                                  = 0;        // set in main as second run of precise_mode
    bool float_is_big_endian                       = false;    // Set in arith_util.h by float_is_big_endian()

    tosa_level_t tosa_level;
    static constexpr tosa_level_t EIGHTK = { 6, 8192, 8192, 256 };
    static constexpr tosa_level_t NONE   = { 0, 0, 0, 0 };
};

#endif

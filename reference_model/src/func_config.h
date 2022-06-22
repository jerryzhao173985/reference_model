
// Copyright (c) 2020-2022, ARM Limited.
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

struct func_config_t
{
    std::string operator_fbs = "tosa.fbs";
    std::string test_desc    = "desc.json";
    std::string flatbuffer_dir = "";
    std::string output_dir = "";
    std::string tosa_file = "";
    std::string ifm_name = "";
    std::string ifm_file = "";
    std::string ofm_name = "";
    std::string ofm_file = "";
    uint32_t eval               = 1;
    uint32_t validate_only      = 0;
    uint32_t output_tensors     = 1;
    uint32_t tosa_profile       = 1;
    uint32_t dump_intermediates = 0;
    std::string fp_format       = "0.5";
};

// Forward declaration
struct func_debug_t;

int func_model_parse_cmd_line(
    func_config_t& func_config, func_debug_t& func_debug, int argc, char** argv, const char* version);
int func_model_parse_flat_config_file(func_config_t*, const char* filename);
void func_model_print_help();

#endif

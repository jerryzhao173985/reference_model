
// Copyright (c) 2020, ARM Limited.
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

// Parameter value structure
#define DEF_UNIT_START(UNIT)                                                                                           \
    struct UNIT##_t                                                                                                    \
    {
#define DEF_UNIT_END(UNIT)                                                                                             \
    }                                                                                                                  \
    UNIT;
#define DEF_OPTION(NAME, DESC, TYPE, FMT, DEFAULT) TYPE NAME;
#define DEF_OPTION_STR(NAME, DESC, LEN, DEFAULT) char NAME[LEN];
#define DEF_UNIT_OPTION(UNIT, NAME, DESC, TYPE, FMT, DEFAULT) TYPE NAME;
#define DEF_UNIT_OPTION_STR(UNIT, NAME, DESC, LEN, DEFAULT) char NAME[LEN];
struct func_config_t
{
#include "func_config.def"
#undef DEF_UNIT_START
#undef DEF_UNIT_END
#undef DEF_OPTION
#undef DEF_OPTION_STR
#undef DEF_UNIT_OPTION
#undef DEF_UNIT_OPTION_STR
};

// Forward declaration
struct func_debug_t;

int func_model_init_config();
int func_model_set_default_config(func_config_t*);
int func_model_config_set_option(func_config_t*, const char* name, const char* value);
int func_model_print_config(func_config_t*, FILE* out);
int func_model_parse_cmd_line(func_config_t*, func_debug_t* func_debug, const int argc, const char** argv, const char* version);
int func_model_parse_flat_config_file(func_config_t*, const char* filename);
int func_model_config_cleanup();
int func_model_config_get_str_option_by_name(func_config_t*, const char* name, char* value, const uint32_t len);
int func_model_config_get_option_by_name(func_config_t*, const char* name, uint64_t* val);
int func_model_print_help(FILE* out);

#endif

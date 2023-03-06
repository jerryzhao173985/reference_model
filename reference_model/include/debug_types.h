
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

/*
 *   Filename:     src/debug_types.h
 *   Description:
 *    Defines fundamental debugger datatypes for the functional model
 */

#include<stdint.h>
#ifndef DEBUG_TYPES_H_
#define DEBUG_TYPES_H_

#ifdef __cplusplus
extern "C"
{
#endif

    // Debug verbosity mask
    typedef enum func_debug_verbosity_e
    {
        DEBUG_VERB_NONE  = 0x00,
        DEBUG_VERB_INFO  = 0x01,    // Informational debugging messages
        DEBUG_VERB_IFACE = 0x02,    // Interface debugging support
        DEBUG_VERB_LOW   = 0x04,    // Low, medium, and high levels of debug printout
        DEBUG_VERB_MED   = 0x08,
        DEBUG_VERB_HIGH  = 0x10
    } func_debug_verbosity_e;

    // Generated debug modes enumeration
    typedef enum func_debug_mode_e
    {
        DEBUG_NONE = 0x0,
#define DEBUG_MODE(NAME, BIT) DEBUG_##NAME = (UINT64_C(1) << BIT),
#include "debug_modes.def"
#undef DEBUG_MODE
        DEBUG_ALL = UINT64_C(0xffffffffffffffff)
    } func_debug_mode_e;

#define DEBUG_INST_ALL UINT64_C(0xffffffffffffffff)

#ifdef __cplusplus
}
#endif

#endif

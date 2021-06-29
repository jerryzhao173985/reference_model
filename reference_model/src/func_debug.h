
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

#ifndef FUNC_DEBUG_H
#define FUNC_DEBUG_H

#include "debug_types.h"
#include <assert.h>
#include <cinttypes>
#include <signal.h>
#include <stdio.h>

void func_print_backtrace(FILE* out, int sig = SIGABRT);

void func_enable_signal_handlers();

// Debug content container
#define WARNING_BUFFER_SIZE 16
#define WARNING_BUFFER_ENTRY_LENGTH 1024

// STRINGIFY2 is needed expand expression passed to STRINGIFY
#define STRINGIFY2(s) #s
#define STRINGIFY(s) STRINGIFY2(s)

// If TRACED_LOG is defined, add file:line to log messages
#if defined(TRACED_LOG)
#define WHERE "@" __FILE__ ":" STRINGIFY(__LINE__)
#else
#define WHERE
#endif

#if defined(COLORIZED_LOG)
#define COL(col, fmt) "\x1b[3" col "m" fmt "\x1b[0m"
#define COL_FATAL(fmt) COL("1;41", fmt)
#define COL_WARN(fmt) COL("1;43", fmt)
#define COL_INFO(fmt) COL("2", fmt)
#define COL_IFACE(fmt) fmt
#define COL_LOW(fmt) COL("35", fmt)
#define COL_MED(fmt) COL("2;33", fmt)
#define COL_HIGH(fmt) COL("2;32", fmt)
#else
#define COL_FATAL(fmt) fmt
#define COL_WARN(fmt) fmt
#define COL_INFO(fmt) fmt
#define COL_IFACE(fmt) fmt
#define COL_LOW(fmt) fmt
#define COL_MED(fmt) fmt
#define COL_HIGH(fmt) fmt
#endif

struct func_debug_t
{
    uint32_t func_debug_verbosity;             // What verbosity level is set? (bitmask)
    uint64_t func_debug_mask;                  // Which units have debugging enabled? (bitmask)
    uint64_t func_debug_inst_mask;             // Which instances have debugging enabled (bitmask)
    uint64_t inst_id;                          // The instance id for multiple model instances
    uint32_t func_suppress_arch_error_mask;    // Which architecture error should be suppressed? (bitmask)
    FILE* func_debug_file;                     // Output file
    uint32_t record_warnings;
    char* warning_buffer[WARNING_BUFFER_SIZE];
    uint32_t warning_buffer_head;    // next unread message
    uint32_t warning_buffer_tail;    // next message to write
    uint32_t is_gzip;
    bool is_output_unbuffered;    // should log files be opened with unbuffered I/O.
};

#ifndef ASSERT
#define ASSERT(COND)                                                                                                   \
    if (!(COND))                                                                                                       \
    {                                                                                                                  \
        fprintf(stderr, COL_FATAL("ASSERTION AT %s:%d %s(): (%s)\n"), __FILE__, __LINE__, __func__, #COND);            \
        func_print_backtrace(stderr);                                                                                  \
        assert(COND);                                                                                                  \
    }
#endif

#ifndef ASSERT_MSG
#define ASSERT_MSG(COND, fmt, ...)                                                                                     \
    if (!(COND))                                                                                                       \
    {                                                                                                                  \
        fprintf(stderr, COL_FATAL("ASSERTION AT %s:%d %s(): (%s)\n"), __FILE__, __LINE__, __func__, #COND);            \
        fprintf(stderr, COL_FATAL(fmt) "\n", ##__VA_ARGS__);                                                           \
        func_print_backtrace(stderr);                                                                                  \
        assert(COND);                                                                                                  \
    }
#endif

#ifndef REQUIRE
#define REQUIRE(COND, fmt, ...)                                                                                        \
    if (!(COND))                                                                                                       \
    {                                                                                                                  \
        fprintf(g_func_debug.func_debug_file, COL_FATAL("REQUIRE() fails AT %s:%d %s(): (%s)\n"), __FILE__, __LINE__,  \
                __func__, #COND);                                                                                      \
        fprintf(g_func_debug.func_debug_file, COL_FATAL(fmt) "\n", ##__VA_ARGS__);                                     \
        this->parent_sgt->setGraphStatus(GraphStatus::TOSA_UNPREDICTABLE);                                             \
    }
#endif

#ifndef ERROR_IF
#define ERROR_IF(COND, fmt, ...)                                                                                       \
    if ((COND))                                                                                                        \
    {                                                                                                                  \
        if (this->parent_sgt->getGraphStatus() != GraphStatus::TOSA_UNPREDICTABLE)                                     \
        {                                                                                                              \
            this->parent_sgt->setGraphStatus(GraphStatus::TOSA_ERROR);                                                 \
        }                                                                                                              \
        fprintf(g_func_debug.func_debug_file, COL_FATAL("ERROR_IF() fails AT %s:%d %s(): (%s)\n"), __FILE__, __LINE__, \
                __func__, #COND);                                                                                      \
        fprintf(g_func_debug.func_debug_file, COL_FATAL(fmt) "\n", ##__VA_ARGS__);                                     \
        this->dumpNode(g_func_debug.func_debug_file);                                                                  \
        func_print_backtrace(g_func_debug.func_debug_file);                                                            \
        return 1;                                                                                                      \
    }
#endif

// Assertion specific to allocating memory
#ifndef ASSERT_MEM
#define ASSERT_MEM(OBJ)                                                                                                \
    if (!(OBJ))                                                                                                        \
    {                                                                                                                  \
        fprintf(stderr, COL_FATAL("ASSERTION AT %s:%d %s(): (" #OBJ "): out of memory\n"), __FILE__, __LINE__,         \
                __func__);                                                                                             \
        func_print_backtrace(stderr);                                                                                  \
        assert(OBJ);                                                                                                   \
    }
#endif

#ifndef FATAL_ERROR
#define FATAL_ERROR(fmt, ...)                                                                                          \
    fprintf(stderr, COL_FATAL("FATAL ERROR AT %s:%d %s():\n"), __FILE__, __LINE__, __func__);                          \
    fprintf(stderr, COL_FATAL(fmt) "\n", ##__VA_ARGS__);                                                               \
    func_print_backtrace(stderr);                                                                                      \
    abort();
#endif

#ifndef SIMPLE_FATAL_ERROR
#define SIMPLE_FATAL_ERROR(fmt, ...)                                                                                   \
    fprintf(stderr, COL_FATAL(fmt) "\n", ##__VA_ARGS__);                                                               \
    exit(1);
#endif

void func_debug_warning(
    func_debug_t* func_debug, const char* file, const char* func, const int line, const char* fmt, ...);
#ifndef WARNING
#define WARNING(...) func_debug_warning(&g_func_debug, __FILE__, __func__, __LINE__, __VA_ARGS__)
#endif

#ifndef WARNING_STDERR
#define WARNING_STDERR(fmt, ...)                                                                                       \
    fprintf(stderr, COL_WARN("WARNING AT %s:%d %s():\n"), __FILE__, __LINE__, __func__);                               \
    fprintf(stderr, COL_WARN(fmt) "\n", ##__VA_ARGS__);
#endif

int func_debug_set_captured_warnings(func_debug_t* func_debug, uint32_t capture);

int func_debug_has_captured_warning(func_debug_t* func_debug);

int func_debug_get_captured_warning(func_debug_t* func_debug, char* buf_ptr, const uint32_t buf_len);

// Is this debug verbosity and unit level enabled?
// Provide compiler hints that this is unlikely
// Two versions, depending on whether DEBUG_INSTANCE_EXPR is defined in a file or not
//
// For .cpp files whose units have discrete instance IDs, define DEBUG_INSTANCE_EXPR to evalute
// to the instance ID variable.  The use of this define in header files is discouraged.

#ifdef DEBUG_INSTANCE_EXPR
// Expression for whether the debugging verbosity + debugging unit is enabled for free-form printouts
#ifdef DEBUG_INSTANCE_EXPR_2
#define DEBUG_ENABLED(VERB, LEVEL)                                                                                     \
    (__builtin_expect((g_func_debug.func_debug_mask == DEBUG_ALL || g_func_debug.func_debug_mask & (DEBUG_##LEVEL)) && \
                          (g_func_debug.func_debug_inst_mask & (uint64_t(1) << (DEBUG_INSTANCE_EXPR))) &&              \
                          (g_func_debug.func_debug_verbosity & (VERB)),                                                \
                      0))
// Debug printing macro
#define DEBUG(VERB, LEVEL, FMT, ...)                                                                                   \
    if (DEBUG_ENABLED(VERB, LEVEL))                                                                                    \
    {                                                                                                                  \
        fprintf(g_func_debug.func_debug_file, "[%d:" #LEVEL "_%02d_%02d" WHERE "]: " FMT "\n",                         \
                (int)g_func_debug.inst_id, (int)(DEBUG_INSTANCE_EXPR), (int)(DEBUG_INSTANCE_EXPR_2), ##__VA_ARGS__);   \
    }

// Prints just the debugging prefix for properly marking free-form printouts
#define DEBUG_PREFIX(LEVEL)                                                                                            \
    fprintf(g_func_debug.func_debug_file, "[%d" #LEVEL "_%02d_%02d" WHERE "]: ", (int)g_func_debug.inst_id,            \
            (int)(DEBUG_INSTANCE_EXPR), (int)(DEBUG_INSTANCE_EXPR_2))

#else    // !DEBUG_INSTANCE_EXPR_2

#define DEBUG_ENABLED(VERB, LEVEL)                                                                                     \
    (__builtin_expect((g_func_debug.func_debug_mask == DEBUG_ALL || g_func_debug.func_debug_mask & (DEBUG_##LEVEL)) && \
                          (g_func_debug.func_debug_inst_mask & (uint64_t(1) << (DEBUG_INSTANCE_EXPR))) &&              \
                          (g_func_debug.func_debug_verbosity & (VERB)),                                                \
                      0))
// Debug printing macro
#define DEBUG(VERB, LEVEL, FMT, ...)                                                                                   \
    if (DEBUG_ENABLED(VERB, LEVEL))                                                                                    \
    {                                                                                                                  \
        fprintf(g_func_debug.func_debug_file, "[%d:" #LEVEL "_%02d" WHERE "]: " FMT "\n", (int)g_func_debug.inst_id,   \
                (int)(DEBUG_INSTANCE_EXPR), ##__VA_ARGS__);                                                            \
    }

// Prints just the debugging prefix for properly marking free-form printouts
#define DEBUG_PREFIX(LEVEL)                                                                                            \
    fprintf(g_func_debug.func_debug_file, "[%d:" #LEVEL "_%02d" WHERE "]: ", (int)g_func_debug.inst_id,                \
            (int)(DEBUG_INSTANCE_EXPR))

#endif    // DEBUG_INSTANCE_EXPR_2

#else    // !DEBUG_INSTANCE_EXPR

// Expression for whether the debugging verbosity + debugging unit is enabled for free-form printouts
#define DEBUG_ENABLED(VERB, LEVEL)                                                                                     \
    (__builtin_expect((g_func_debug.func_debug_mask == DEBUG_ALL || g_func_debug.func_debug_mask & (DEBUG_##LEVEL)) && \
                          (g_func_debug.func_debug_verbosity & (VERB)),                                                \
                      0))
// Debug printing macro
#define DEBUG(VERB, LEVEL, FMT, ...)                                                                                   \
    if (DEBUG_ENABLED(VERB, LEVEL))                                                                                    \
    {                                                                                                                  \
        fprintf(g_func_debug.func_debug_file, "[%d:" #LEVEL WHERE "]: " FMT "\n", (int)g_func_debug.inst_id,           \
                ##__VA_ARGS__);                                                                                        \
    }

// Prints just the debugging prefix for properly marking free-form printouts
#define DEBUG_PREFIX(LEVEL) fprintf(g_func_debug.func_debug_file, "[" #LEVEL WHERE "]: ")

#endif

// Macros for different verbosity levels
#define DEBUG_INFO(LEVEL, FMT, ...) DEBUG(DEBUG_VERB_INFO, LEVEL, COL_INFO(FMT), ##__VA_ARGS__)
#define DEBUG_IFACE(LEVEL, FMT, ...) DEBUG(DEBUG_VERB_IFACE, LEVEL, COL_IFACE(FMT), ##__VA_ARGS__)
#define DEBUG_LOW(LEVEL, FMT, ...) DEBUG(DEBUG_VERB_LOW, LEVEL, COL_LOW(FMT), ##__VA_ARGS__)
#define DEBUG_MED(LEVEL, FMT, ...) DEBUG(DEBUG_VERB_MED, LEVEL, COL_MED(FMT), ##__VA_ARGS__)
#define DEBUG_HIGH(LEVEL, FMT, ...) DEBUG(DEBUG_VERB_HIGH, LEVEL, COL_HIGH(FMT), ##__VA_ARGS__)

int func_init_debug(func_debug_t*, uint64_t inst_id);
int func_fini_debug(func_debug_t*);
int func_debug_set_file(func_debug_t*, const char* filename);
void func_debug_set_mask(func_debug_t*, const char* str);
void func_debug_set_mask(func_debug_t*, const uint64_t mask);
void func_debug_print_masks(FILE* out);
void func_debug_set_verbosity(func_debug_t*, const char* str);
void func_debug_set_verbosity(func_debug_t*, const uint32_t verb);
void func_debug_set_suppress_arch_error_mask(func_debug_t*, const uint32_t suppress);
void func_debug_set_inst_mask(func_debug_t*, const char* mask);
void func_debug_set_inst_mask(func_debug_t*, const uint64_t mask);
void func_debug_set_output_unbuffered(func_debug_t*, const bool is_unbuffered);

#endif

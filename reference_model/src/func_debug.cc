
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

#include <ctype.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#ifndef _MSC_VER
#include <execinfo.h>
#include <sys/prctl.h>
#include <sys/ptrace.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include "func_debug.h"

#define MAX_FRAMES 100

#ifndef _MSC_VER
pid_t func_print_backtrace_helper(int num_tries, int sig);
#endif

void func_print_backtrace(FILE* out, int sig)
{
#ifndef _MSC_VER
    for (int i = 0; i < 2; i++)
    {
        const pid_t child_pid = func_print_backtrace_helper(i, sig);
        if (child_pid < 0)
        {
            perror("Backtrace generation failed on fork");
            break;
        }

        int status = 0;
        waitpid(child_pid, &status, 0);
        if (WEXITSTATUS(status) == 0)
        {
            break;
        }
    }
#endif
}

#ifndef _MSC_VER
pid_t func_print_backtrace_helper(int num_tries, int sig)
{
    const pid_t child_pid = fork();

    if (child_pid)
    {
        return 0;
    }

    const pid_t ppid = getppid();

    printf("Attaching debugger to pid %d\n", ppid);
    // Check if we're in a debugger
    if (ptrace(PTRACE_ATTACH, ppid, 0, 0) == 0)
    {
        // If we reach this point, no debugger is present
        // Undo effects of PTRACE_ATTACH
        waitpid(ppid, NULL, 0);
        ptrace(PTRACE_CONT, 0, 0, 0);
        ptrace(PTRACE_DETACH, ppid, 0, 0);

        dup2(STDERR_FILENO, STDOUT_FILENO);

        char parent_pid[20];
        snprintf(parent_pid, sizeof(parent_pid), "attach %d", ppid);
        fprintf(stdout, "Caught signal %d (%s)\n", sig, strsignal(sig));

        execlp("gdb", "gdb", "--batch", "-n", "-ex",
               // Don't print startup messages for each thread
               "-ex", "set print thread-events off", "-ex", parent_pid,
               // Turn off pagination
               "-ex", "set height 0",
               // Print a backtrace for the current thread
               "-ex", "thread $_thread", "-ex", "bt",
               // Print a backtrace for the main thread (uncomment the next two lines, if desired)
               //"-ex", "thread 1",
               //"-ex", "bt",
               // Print a backtrace for all thread (TMI)
               //"-ex", "thread apply all bt",
               NULL);

        // If we reach this point, it is bad. Attempt to print an error before exiting.
        perror("Backtrace generation failed to invoke gdb");
        exit(1);
    }

    // Debugger present.  Exit here.
    exit(0);

    return 0;
}
#endif

void func_backtrace_signal_handler(int sig)
{
    func_print_backtrace(NULL, sig);
    exit(1);
}

// Note: this overwrites other signal handlers.  May want to make this
// more friendly sometime
void func_enable_signal_handlers()
{
    static const int sig_list[] = { SIGABRT, SIGSEGV, SIGILL, SIGFPE };

    if (getenv("FUNC_NO_SIG_HANDLERS"))
    {
        return;
    }

    for (size_t i = 0; i < sizeof(sig_list) / sizeof(int); i++)
    {
        struct sigaction act;

        bzero(&act, sizeof(act));
        act.sa_handler = func_backtrace_signal_handler;

        if (sigaction(sig_list[i], &act, NULL))
        {
            perror("Error calling sigaction");
        }
    }
}

const char* func_debug_mode_str_table[] = {
#define DEBUG_MODE(NAME, BIT) #NAME,
#include "debug_modes.def"
#undef DEBUG_MODE
};

#define DEBUG_MASK_COUNT (sizeof(func_debug_mode_str_table) / sizeof(const char*))

const char* func_debug_verbosity_str_table[] = { "NONE", "INFO", "IFACE", "LOW", "MED", "HIGH" };

const uint32_t func_debug_verbosity_mask_table[] = { DEBUG_VERB_NONE, DEBUG_VERB_INFO, DEBUG_VERB_IFACE,
                                                     DEBUG_VERB_LOW,  DEBUG_VERB_MED,  DEBUG_VERB_HIGH };

#define DEBUG_VERBOSITY_COUNT (sizeof(func_debug_verbosity_str_table) / sizeof(const char*))

// Initialize the debug mode
int func_init_debug(func_debug_t* func_debug, uint64_t inst_id)
{
    // Set the default debug settings
    bzero(func_debug, sizeof(func_debug_t));
    func_debug_set_mask(func_debug, DEBUG_NONE);
    func_debug_set_verbosity(func_debug, DEBUG_VERB_NONE);
    func_debug_set_inst_mask(func_debug, DEBUG_INST_ALL);
    func_debug->func_debug_file = stderr;
    func_debug_set_captured_warnings(func_debug, 0);
    func_debug_set_output_unbuffered(func_debug, false);
    func_debug->inst_id = inst_id;

    return 0;
}

int func_fini_debug(func_debug_t* func_debug)
{
    if (func_debug->record_warnings)
    {
        func_debug_set_captured_warnings(func_debug, 0);
    }

#ifndef _FUNC_INCLUDE_WINDOWS_SUPPORT_H
    if (func_debug->is_gzip && func_debug->func_debug_file)
    {
        pclose(func_debug->func_debug_file);
        func_debug->func_debug_file = NULL;
    }
#endif

    return 0;
}

int func_debug_set_file(func_debug_t* func_debug, const char* filename)
{
    int filenameLen = strlen(filename);

    // Open the debug output file
    ASSERT(filename != NULL);
#ifndef _FUNC_INCLUDE_WINDOWS_SUPPORT_H
    if (filenameLen > 3 && strcmp(filename + filenameLen - 3, ".gz") == 0)
    {
        char cmd[256];

        snprintf(cmd, sizeof(cmd), "gzip > %s", filename);
        func_debug->func_debug_file = popen(cmd, "w");
        func_debug->is_gzip         = 1;
    }
    else
    {
#else
    {
#endif
        func_debug->func_debug_file = fopen(filename, "w");
    }

    if (!func_debug->func_debug_file)
    {
        perror(NULL);
        FATAL_ERROR("Cannot open debug output file: %s\n", filename);
        return 1;
    }
    if (func_debug->is_output_unbuffered)
    {
        setvbuf(func_debug->func_debug_file, nullptr, _IONBF, 0);
    }

    return 0;
}

void func_debug_set_verbosity(func_debug_t* func_debug, const char* str)
{
    if (!strcasecmp(str, "RESET"))
    {
        func_debug_set_verbosity(func_debug, DEBUG_VERB_NONE);
        return;
    }

    for (size_t i = 0; i < DEBUG_VERBOSITY_COUNT; i++)
    {
        if (!strcasecmp(str, func_debug_verbosity_str_table[i]))
        {
            func_debug_set_verbosity(func_debug, func_debug_verbosity_mask_table[i]);
            return;
        }
    }

    FATAL_ERROR("Invalid debug verbosity: %s", str);
}

void func_debug_set_verbosity(func_debug_t* func_debug, const uint32_t verb)
{
    uint32_t new_mask = verb;

    switch (verb)
    {
        case DEBUG_VERB_NONE:
            new_mask = DEBUG_VERB_NONE;
            break;
        case DEBUG_VERB_INFO:
            new_mask = DEBUG_VERB_INFO;
            break;
        case DEBUG_VERB_IFACE:
            new_mask = DEBUG_VERB_IFACE;
            break;
        case DEBUG_VERB_HIGH:
            new_mask |= DEBUG_VERB_HIGH;
            // Intentional fallthrough
        case DEBUG_VERB_MED:
            new_mask |= DEBUG_VERB_MED;
            // Intentional fallthrough
        case DEBUG_VERB_LOW:
            new_mask |= DEBUG_VERB_LOW;
            new_mask |= DEBUG_VERB_INFO;
            new_mask |= DEBUG_VERB_IFACE;
            break;
    }

    func_debug->func_debug_verbosity = new_mask;
}

void func_debug_set_suppress_arch_error_mask(func_debug_t* func_debug, const uint32_t suppress)
{
    func_debug->func_suppress_arch_error_mask = suppress;
}

void func_debug_set_mask(func_debug_t* func_debug, const uint64_t mask)
{
    if (mask == DEBUG_NONE)
        func_debug->func_debug_mask = mask;
    else
        func_debug->func_debug_mask |= mask;

    // Set a minimum verbosity level
    if (func_debug->func_debug_verbosity == DEBUG_VERB_NONE)
        func_debug->func_debug_verbosity = DEBUG_VERB_INFO;
}

void func_debug_set_inst_mask(func_debug_t* func_debug, const char* mask)
{
    uint64_t val;

    val = strtoul(mask, NULL, 0);

    return func_debug_set_inst_mask(func_debug, val);
}

void func_debug_set_inst_mask(func_debug_t* func_debug, const uint64_t mask)
{
    if (mask == 0)
        func_debug->func_debug_inst_mask = DEBUG_INST_ALL;
    else
        func_debug->func_debug_inst_mask = mask;
}

void func_debug_set_mask(func_debug_t* func_debug, const char* str)
{
    if (!strcasecmp(str, "all"))
    {
        func_debug_set_mask(func_debug, UINT64_MAX - 1);
        return;
    }

    size_t i;
    for (i = 0; i < DEBUG_MASK_COUNT; i++)
    {
        if (!strcasecmp(str, func_debug_mode_str_table[i]))
        {
            func_debug_set_mask(func_debug, 1ULL << i);
            return;
        }
    }

    func_debug_print_masks(stderr);

    FATAL_ERROR("Invalid debug mask: %s", str);
}

void func_debug_print_masks(FILE* out)
{
    uint32_t i;

    fprintf(out, "Available debug masks:\n");

    for (i = 0; i < DEBUG_MASK_COUNT; i++)
    {
        fprintf(out, "[%d] %s\n", i, func_debug_mode_str_table[i]);
    }
}

void func_debug_set_output_unbuffered(func_debug_t* func_debug, const bool is_unbuffered)
{
    func_debug->is_output_unbuffered = is_unbuffered;
}

// Print warnings to the debug file or optionally store them in a buffer instead
// Note that the buffer is circular and can be overwritten if enough messages are
// written before removing a warning from the front.
void func_debug_warning(
    func_debug_t* func_debug, const char* file, const char* func, const int line, const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);

    if (func_debug->record_warnings)
    {
        // Record to the circular buffer
        uint32_t len;

        len = snprintf(func_debug->warning_buffer[func_debug->warning_buffer_tail], WARNING_BUFFER_ENTRY_LENGTH,
                       "WARNING AT %s:%d %s(): ", file, line, func);
        vsnprintf(func_debug->warning_buffer[func_debug->warning_buffer_tail] + len, WARNING_BUFFER_ENTRY_LENGTH - len,
                  fmt, args);
        func_debug->warning_buffer_tail = (func_debug->warning_buffer_tail + 1) % WARNING_BUFFER_SIZE;
    }
    else
    {
        // Print to the debug file (e.g., stderr)
        fprintf(func_debug->func_debug_file, "WARNING AT %s:%d %s():\n", file, line, func);
        vfprintf(func_debug->func_debug_file, fmt, args);
        fprintf(func_debug->func_debug_file, "\n");
    }
    va_end(args);
}

// Initialize the warning buffer capture
int func_debug_set_captured_warnings(func_debug_t* func_debug, uint32_t capture)
{
    uint32_t i;
    func_debug->record_warnings = capture;
    if (capture)
    {
        func_debug->warning_buffer_head = 0;
        func_debug->warning_buffer_tail = 0;

        for (i = 0; i < WARNING_BUFFER_SIZE; i++)
        {
            func_debug->warning_buffer[i] = (char*)calloc(1, WARNING_BUFFER_ENTRY_LENGTH);
        }
    }
    else
    {
        for (i = 0; i < WARNING_BUFFER_SIZE; i++)
        {
            if (func_debug->warning_buffer[i])
            {
                free(func_debug->warning_buffer[i]);
                func_debug->warning_buffer[i] = NULL;
            }
        }
    }

    return 0;
}

int func_debug_has_captured_warning(func_debug_t* func_debug)
{
    if (func_debug->record_warnings && func_debug->warning_buffer_head != func_debug->warning_buffer_tail)
        return 1;
    else
        return 0;
}

int func_debug_get_captured_warning(func_debug_t* func_debug, char* buf_ptr, const uint32_t buf_len)
{
    if (!func_debug_has_captured_warning(func_debug))
        return 1;

    strncpy(buf_ptr, func_debug->warning_buffer[func_debug->warning_buffer_head], buf_len);

    func_debug->warning_buffer_head = (func_debug->warning_buffer_head + 1) % WARNING_BUFFER_SIZE;

    return 0;
}

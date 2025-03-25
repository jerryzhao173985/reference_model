
// Copyright (c) 2020-2023, 2025 ARM Limited.
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
#include <sstream>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#ifndef _WIN32
#include <execinfo.h>
#if !defined(__APPLE__) && !defined(__MACH__)
#include <sys/prctl.h>
#endif
#include <sys/ptrace.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include "func_debug.h"

#define MAX_FRAMES 100

static bool str_case_equal(const std::string& a, const std::string& b)
{
    return std::equal(a.begin(), a.end(), b.begin(), b.end(),
                      [](char ac, char bc) { return tolower(ac) == tolower(bc); });
}

#if !defined(_WIN32) && !defined(__APPLE__) && !defined(__MACH__)
pid_t func_print_backtrace_helper(int num_tries, int sig);
#endif

void func_print_backtrace(FILE* out, int sig)
{
#if !defined(_WIN32) && !defined(__APPLE__) && !defined(__MACH__)
    if (getenv("TOSA_MODEL_NO_BACKTRACE"))
        return;

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

#if !defined(_WIN32) && !defined(__APPLE__) && !defined(__MACH__)
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

        std::string parent_pid_str = "attach " + std::to_string(ppid);
        fprintf(stdout, "Caught signal %d (%s)\n", sig, strsignal(sig));

        execlp("gdb", "gdb", "--batch", "-n", "-ex",
               // Don't print startup messages for each thread
               "-ex", "set print thread-events off", "-ex", parent_pid_str.c_str(),
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
#if !defined(_WIN32) && !defined(__APPLE__) && !defined(__MACH__)
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
#endif

static const std::vector<std::pair<std::string, uint32_t>> func_debug_verbosity_table = {
    { "NONE", DEBUG_VERB_NONE }, { "INFO", DEBUG_VERB_INFO }, { "IFACE", DEBUG_VERB_IFACE },
    { "LOW", DEBUG_VERB_LOW },   { "MED", DEBUG_VERB_MED },   { "HIGH", DEBUG_VERB_HIGH }
};

// Initialize the debug mode
int func_debug_t::init_debug(uint64_t inst_id)
{
    // Set the default debug settings
    set_mask(static_cast<uint64_t>(DEBUG_NONE));
    set_verbosity(DEBUG_VERB_NONE);
    set_inst_mask(DEBUG_INST_ALL);
    func_debug_file = stderr;
    this->inst_id   = inst_id;

    return 0;
}

int func_debug_t::fini_debug()
{
    return 0;
}

int func_debug_t::set_file(const std::string& filename)
{
    // Open the debug output file
    func_debug_file = fopen(filename.c_str(), "w");

    if (!func_debug_file)
    {
        perror(NULL);
        FATAL_ERROR("Cannot open debug output file: %s\n", filename.c_str());
        return 1;
    }
    if (is_output_unbuffered)
    {
        setvbuf(func_debug_file, nullptr, _IONBF, 0);
    }

    return 0;
}

void func_debug_t::set_verbosity(const std::string& str)
{
    for (auto& verb : func_debug_verbosity_table)
    {
        if (str_case_equal(str, verb.first))
        {
            set_verbosity(verb.second);
            return;
        }
    }

    FATAL_ERROR("Invalid debug verbosity: %s", str.c_str());
}

void func_debug_t::set_verbosity(const uint32_t verb)
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
            [[fallthrough]];
        case DEBUG_VERB_MED:
            new_mask |= DEBUG_VERB_MED;
            [[fallthrough]];
        case DEBUG_VERB_LOW:
            new_mask |= DEBUG_VERB_LOW;
            new_mask |= DEBUG_VERB_INFO;
            new_mask |= DEBUG_VERB_IFACE;
            break;
    }

    func_debug_verbosity = new_mask;
}

void func_debug_t::set_mask(const uint64_t mask)
{
    if (mask == DEBUG_NONE)
        func_debug_mask = mask;
    else
        func_debug_mask |= mask;

    // Set a minimum verbosity level
    if (func_debug_verbosity == DEBUG_VERB_NONE)
        func_debug_verbosity = DEBUG_VERB_INFO;
}

void func_debug_t::set_inst_mask(const char* mask)
{
    uint64_t val;

    val = strtoul(mask, NULL, 0);

    return set_inst_mask(val);
}

void func_debug_t::set_inst_mask(const uint64_t mask)
{
    if (mask == 0)
        func_debug_inst_mask = DEBUG_INST_ALL;
    else
        func_debug_inst_mask = mask;
}

std::vector<std::pair<std::string, int>> debug_str_table = {
#define DEBUG_MODE(NAME, BIT) { #NAME, DEBUG_##NAME },
#include "debug_modes.def"
#undef DEBUG_MODE
};

void func_debug_t::set_mask(const std::string& str)
{
    if (str == "ALL")
    {
        set_mask(UINT64_MAX - 1);
        return;
    }
    for (auto& mode : debug_str_table)
    {
        if (mode.first == str)
        {
            set_mask(static_cast<uint64_t>(mode.second));
            return;
        }
    }
    print_masks(stderr);

    FATAL_ERROR("Invalid debug mask: %s", str.c_str());
}

void func_debug_t::print_masks(FILE* out)
{
    fprintf(out, "Available debug masks:\n");
    for (auto& mode : debug_str_table)
    {
        fprintf(out, "[%d] %s\n", mode.second, mode.first.c_str());
    }
}

void func_debug_t::set_output_unbuffered(const bool is_unbuffered)
{
    is_output_unbuffered = is_unbuffered;
}

// Print warnings to the debug file or optionally store them in a buffer instead
// Note that the buffer is circular and can be overwritten if enough messages are
// written before removing a warning from the front.
void func_debug_warning(
    func_debug_t* func_debug, const char* file, const char* func, const int line, const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);

    // Print to the debug file (e.g., stderr)
    fprintf(func_debug->func_debug_file, "WARNING AT %s:%d %s():\n", file, line, func);
    vfprintf(func_debug->func_debug_file, fmt, args);
    fprintf(func_debug->func_debug_file, "\n");

    va_end(args);
}

std::string func_debug_t::get_debug_mask_help_string()
{
    std::string rval = "Set debug mask. Valid values are: ";
    for (auto& mask : debug_str_table)
    {
        rval += mask.first + " ";
    }
    rval += "ALL";
    return rval;
}

std::string func_debug_t::get_debug_verbosity_help_string()
{
    std::string rval = "Set logging level. Valid values are: ";
    for (auto& verb : func_debug_verbosity_table)
    {
        rval += verb.first + " ";
    }
    return rval;
}

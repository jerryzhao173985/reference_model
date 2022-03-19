
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

#include <ctype.h>
#include <signal.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#include "func_config.h"
#include "func_debug.h"

#define MAX_NAME_LEN 128
#define MAX_DESC_LEN 128

#ifndef ARG_ERROR
#define ARG_ERROR(...)                                                                                                 \
    fprintf(stderr, "ERROR: ");                                                                                        \
    fprintf(stderr, __VA_ARGS__);                                                                                      \
    fprintf(stderr, "\n");                                                                                             \
    return 1;
#endif

// Parameter base name string table
const char* config_base_name_table[] = {
#define DEF_UNIT_START(UNIT)
#define DEF_UNIT_END(UNIT)
#define DEF_OPTION(NAME, DESC, TYPE, FMT, DEFAULT) #NAME,
#define DEF_OPTION_STR(NAME, DESC, LEN, DEFAULT) #NAME,
#define DEF_UNIT_OPTION(UNIT, NAME, DESC, TYPE, FMT, DEFAULT) #NAME,
#define DEF_UNIT_OPTION_STR(UNIT, NAME, DESC, LEN, DEFAULT) #NAME,
#include "func_config.def"
#undef DEF_UNIT_START
#undef DEF_UNIT_END
#undef DEF_OPTION
#undef DEF_OPTION_STR
#undef DEF_UNIT_OPTION_STR
#undef DEF_UNIT_OPTION
};

// Parameter description table
const char* config_param_desc_table[] = {
#define DEF_UNIT_START(UNIT)
#define DEF_UNIT_END(UNIT)
#define DEF_OPTION(NAME, DESC, TYPE, FMT, DEFAULT) #DESC,
#define DEF_OPTION_STR(NAME, DESC, LEN, DEFAULT) #DESC,
#define DEF_UNIT_OPTION(UNIT, NAME, DESC, TYPE, FMT, DEFAULT) #DESC,
#define DEF_UNIT_OPTION_STR(UNIT, NAME, DESC, LEN, DEFAULT) #DESC,
#include "func_config.def"
#undef DEF_UNIT_START
#undef DEF_UNIT_END
#undef DEF_OPTION
#undef DEF_UNIT_OPTION
#undef DEF_OPTION_STR
#undef DEF_UNIT_OPTION_STR
};

// String table and enum for the option hierarchy level/sub-levels
// (no leaf options).  Attribute at the top level have "BASE" as their
// enum value and an empty string for the value.
const char* config_hier_str_table[] = {
    "",
#define DEF_UNIT_START(UNIT) #UNIT,
#define DEF_UNIT_END(UNIT)                                    /**/
#define DEF_OPTION(NAME, DESC, TYPE, FMT, DEFAULT)            /**/
#define DEF_OPTION_STR(NAME, DESC, LEN, DEFAULT)              /**/
#define DEF_UNIT_OPTION(UNIT, NAME, DESC, TYPE, FMT, DEFAULT) /**/
#define DEF_UNIT_OPTION_STR(UNIT, NAME, DESC, LEN, DEFAULT)   /**/
#include "func_config.def"
#undef DEF_UNIT_START
#undef DEF_UNIT_END
#undef DEF_OPTION
#undef DEF_UNIT_OPTION
#undef DEF_OPTION_STR
#undef DEF_UNIT_OPTION_STR
};

typedef enum config_hier_enum_t
{
    BASE,
#define DEF_UNIT_START(UNIT) CURRENT_UNIT,
#define DEF_UNIT_END(UNIT)                                    /**/
#define DEF_OPTION(NAME, DESC, TYPE, FMT, DEFAULT)            /**/
#define DEF_OPTION_STR(NAME, DESC, LEN, DEFAULT)              /**/
#define DEF_UNIT_OPTION(UNIT, NAME, DESC, TYPE, FMT, DEFAULT) /**/
#define DEF_UNIT_OPTION_STR(UNIT, NAME, DESC, LEN, DEFAULT)   /**/
#include "func_config.def"
#undef DEF_UNIT_START
#undef DEF_UNIT_END
#undef DEF_OPTION
#undef DEF_UNIT_OPTION
#undef DEF_OPTION_STR
#undef DEF_UNIT_OPTION_STR

    MAX_CONFIG_HIER
} config_hier_enum_t;

// Mapping from a leaf parameter index to the
// position in the hierarchy.
config_hier_enum_t config_hierarchy_map[] = {
#define DEF_UNIT_START(UNIT)
#define DEF_UNIT_END(UNIT)
#define DEF_OPTION(NAME, DESC, TYPE, FMT, DEFAULT) BASE,
#define DEF_OPTION_STR(NAME, DESC, LEN, DEFAULT) BASE,
#define DEF_UNIT_OPTION(UNIT, NAME, DESC, TYPE, FMT, DEFAULT) CURRENT_UNIT,
#define DEF_UNIT_OPTION_STR(UNIT, NAME, DESC, LEN, DEFAULT) CURRENT_UNIT,
#include "func_config.def"
#undef DEF_UNIT_START
#undef DEF_UNIT_END
#undef DEF_OPTION
#undef DEF_UNIT_OPTION
#undef DEF_OPTION_STR
#undef DEF_UNIT_OPTION_STR
};

#define CONFIG_PARAMETER_COUNT (sizeof(config_hierarchy_map) / sizeof(config_hier_enum_t))

// Dynamically generated at initialization
char** config_param_str_table = nullptr;

// Initialize the configuration data structures
int func_model_init_config()
{
    // Initialize string table (builds the hierarchical names)
    config_param_str_table = (char**)calloc(CONFIG_PARAMETER_COUNT, sizeof(char*));
    ASSERT_MEM(config_param_str_table);

    for (uint32_t i = 0; i < CONFIG_PARAMETER_COUNT; i++)
    {
        size_t len = strlen(config_base_name_table[i]) + 1;
        if (config_hierarchy_map[i] != BASE)
        {
            ASSERT_MSG(config_hierarchy_map[i] <= MAX_CONFIG_HIER,
                       "Configuration parameter\'s hierarchy is out of bounds");
            len += strlen(config_hier_str_table[config_hierarchy_map[i]]) + 1;
        }
        config_param_str_table[i] = (char*)calloc(len, 1);
        ASSERT_MEM(config_param_str_table[i]);
        ASSERT_MSG(len < MAX_NAME_LEN, "option expanded name is too long: %s", config_base_name_table[i]);

        if (config_hierarchy_map[i] != BASE)
        {
            snprintf(config_param_str_table[i], len, "%s.%s", config_hier_str_table[config_hierarchy_map[i]],
                     config_base_name_table[i]);
        }
        else
        {
            snprintf(config_param_str_table[i], len, "%s", config_base_name_table[i]);
        }
    }

    return 0;
}

int func_model_set_default_config(func_config_t* func_config)
{
    // Set default values in the global configuration data structure
    bzero(func_config, sizeof(*func_config));

#define DEF_UNIT_START(UNIT)
#define DEF_UNIT_END(UNIT)
#define DEF_OPTION(NAME, DESC, TYPE, FMT, DEFAULT) func_config->NAME = (DEFAULT);
#define DEF_OPTION_STR(NAME, DESC, LEN, DEFAULT) strncpy(func_config->NAME, (DEFAULT), (LEN)-1);
#define DEF_UNIT_OPTION(UNIT, NAME, DESC, TYPE, FMT, DEFAULT) func_config->UNIT.NAME = (DEFAULT);
#define DEF_UNIT_OPTION_STR(UNIT, NAME, DESC, LEN, DEFAULT) strncpy(func_config->UNIT.NAME, (DEFAULT), (LEN)-1);
#include "func_config.def"
#undef DEF_UNIT_START
#undef DEF_UNIT_END
#undef DEF_OPTION
#undef DEF_UNIT_OPTION
#undef DEF_OPTION_STR
#undef DEF_UNIT_OPTION_STR

    return 0;
}

int func_model_config_cleanup()
{
    uint32_t i;

    if (!config_param_str_table)
        return 1;

    for (i = 0; i < CONFIG_PARAMETER_COUNT; i++)
    {
        free(config_param_str_table[i]);
    }

    free(config_param_str_table);
    config_param_str_table = nullptr;

    return 0;
}

int func_model_config_set_option(func_config_t* func_config, const char* name, const char* value)
{
    // Increment an index variable on each parameter position
    // so that we can index both the position struct through the macro and the
    // array of parameter names through a simple array of strings.
    int param_idx = 0;
    char* endptr;

    // TODO: does not handle strings yet.  Can set magic values on FMT to
    // choose a string copy vs strtoull
#define DEF_UNIT_START(UNIT)
#define DEF_UNIT_END(UNIT)
#define DEF_OPTION(NAME, DESC, TYPE, FMT, DEFAULT)                                                                     \
    if (!strcmp(config_param_str_table[param_idx], name))                                                              \
    {                                                                                                                  \
        func_config->NAME = (uint64_t)strtoll(value, &endptr, 0);                                                      \
        if (endptr == value)                                                                                           \
        {                                                                                                              \
            ARG_ERROR("Cannot parse option: %s = %s", name, value);                                                    \
        }                                                                                                              \
        return 0;                                                                                                      \
    }                                                                                                                  \
    param_idx++;

#define DEF_OPTION_STR(NAME, DESC, LEN, DEFAULT)                                                                       \
    if (!strcmp(config_param_str_table[param_idx], name))                                                              \
    {                                                                                                                  \
        if (strlen(value) >= LEN)                                                                                      \
        {                                                                                                              \
            ARG_ERROR("Option value is too long: %s = %s", name, value);                                               \
        }                                                                                                              \
        strncpy(func_config->NAME, value, (LEN)-1);                                                                    \
        return 0;                                                                                                      \
    }                                                                                                                  \
    param_idx++;

#define DEF_UNIT_OPTION(UNIT, NAME, DESC, TYPE, FMT, DEFAULT)                                                          \
    if (!strcmp(config_param_str_table[param_idx], name))                                                              \
    {                                                                                                                  \
        func_config->UNIT.NAME = (uint64_t)strtoll(value, &endptr, 0);                                                 \
        if (endptr == value)                                                                                           \
        {                                                                                                              \
            ARG_ERROR("Cannot parse option: %s = %s", name, value);                                                    \
        }                                                                                                              \
        return 0;                                                                                                      \
    }                                                                                                                  \
    param_idx++;

#define DEF_UNIT_OPTION_STR(UNIT, NAME, DESC, LEN, DEFAULT)                                                            \
    if (!strcmp(config_param_str_table[param_idx], name))                                                              \
    {                                                                                                                  \
        if (strlen(value) >= LEN)                                                                                      \
        {                                                                                                              \
            ARG_ERROR("Option value is too long: %s = %s", name, value);                                               \
        }                                                                                                              \
        strncpy(func_config->UNIT.NAME, value, (LEN)-1);                                                               \
        return 0;                                                                                                      \
    }                                                                                                                  \
    param_idx++;

#include "func_config.def"
#undef DEF_UNIT_START
#undef DEF_UNIT_END
#undef DEF_OPTION
#undef DEF_UNIT_OPTION
#undef DEF_OPTION_STR
#undef DEF_UNIT_OPTION_STR

    // No match!
    ARG_ERROR("Cannot find option: %s", name);

    return 1;
}

int func_model_config_get_option_by_name(func_config_t* func_config, const char* name, uint64_t* val)
{
    // Increment an index variable on each parameter position
    // so that we can index both the position struct through the macro and the
    // array of parameter names through a simple array of strings.
    int param_idx = 0;

#define DEF_UNIT_START(UNIT)
#define DEF_UNIT_END(UNIT)

#define DEF_OPTION_STR(NAME, DESC, LEN, DEFAULT) param_idx++;

#define DEF_UNIT_OPTION_STR(UNIT, NAME, DESC, FMT, DEFAULT) param_idx++;

#define DEF_OPTION(NAME, DESC, TYPE, FMT, DEFAULT)                                                                     \
    if (!strcmp(config_param_str_table[param_idx], name))                                                              \
    {                                                                                                                  \
        *val = func_config->NAME;                                                                                      \
        return 0;                                                                                                      \
    }                                                                                                                  \
    param_idx++;

#define DEF_UNIT_OPTION(UNIT, NAME, DESC, TYPE, FMT, DEFAULT)                                                          \
    if (!strcmp(config_param_str_table[param_idx], name))                                                              \
    {                                                                                                                  \
        *val = func_config->UNIT.NAME;                                                                                 \
        return 0;                                                                                                      \
    }                                                                                                                  \
    param_idx++;

#include "func_config.def"
#undef DEF_UNIT_START
#undef DEF_UNIT_END
#undef DEF_OPTION
#undef DEF_UNIT_OPTION
#undef DEF_OPTION_STR
#undef DEF_UNIT_OPTION_STR
    // No match!
    return 1;
}
int func_model_config_get_str_option_by_name(func_config_t* func_config,
                                             const char* name,
                                             char* value,
                                             const uint32_t len)
{
    // Increment an index variable on each parameter position
    // so that we can index both the position struct through the macro and the
    // array of parameter names through a simple array of strings.
    int param_idx = 0;

#define DEF_UNIT_START(UNIT)
#define DEF_UNIT_END(UNIT)
#define DEF_OPTION_STR(NAME, DESC, LEN, DEFAULT)                                                                       \
    if (!strcmp(config_param_str_table[param_idx], name))                                                              \
    {                                                                                                                  \
        strncpy(value, func_config->NAME, len - 1);                                                                    \
        return 0;                                                                                                      \
    }                                                                                                                  \
    param_idx++;

#define DEF_UNIT_OPTION_STR(UNIT, NAME, DESC, LEN, DEFAULT)                                                            \
    if (!strcmp(config_param_str_table[param_idx], name))                                                              \
    {                                                                                                                  \
        strncpy(value, func_config->UNIT.NAME, len - 1);                                                               \
        return 0;                                                                                                      \
    }                                                                                                                  \
    param_idx++;

#define DEF_OPTION(NAME, DESC, TYPE, FMT, DEFAULT) param_idx++;

#define DEF_UNIT_OPTION(UNIT, NAME, DESC, TYPE, FMT, DEFAULT) param_idx++;

#include "func_config.def"
#undef DEF_UNIT_START
#undef DEF_UNIT_END
#undef DEF_OPTION
#undef DEF_UNIT_OPTION
#undef DEF_OPTION_STR
#undef DEF_UNIT_OPTION_STR
    // No match!
    return 1;
}

int func_config_print_config_help(FILE* out)
{
    fprintf(out, "%-40s %s\n", "Option", "Description");
    fprintf(out, "%-40s %s\n", "------", "-----------");

    for (uint32_t i = 0; i < CONFIG_PARAMETER_COUNT; i++)
    {
        fprintf(out, "-C%-40s %s\n", config_param_str_table[i], config_param_desc_table[i]);
    }

    fprintf(out, "\n");

    return 0;
}

int func_model_print_config(func_config_t* func_config, FILE* out)
{
#define DEF_UNIT_START(UNIT)
#define DEF_UNIT_END(UNIT)
#define DEF_OPTION(NAME, DESC, TYPE, FMT, DEFAULT) fprintf(out, "%-40s = " FMT "\n", #NAME, func_config->NAME);
#define DEF_UNIT_OPTION(UNIT, NAME, DESC, TYPE, FMT, DEFAULT)                                                          \
    fprintf(out, "%-40s = " FMT "\n", #UNIT "." #NAME, func_config->UNIT.NAME);
#define DEF_OPTION_STR(NAME, DESC, LEN, DEFAULT) fprintf(out, "%-40s = %s\n", #NAME, func_config->NAME);
#define DEF_UNIT_OPTION_STR(UNIT, NAME, DESC, LEN, DEFAULT)                                                            \
    fprintf(out, "%-40s = %s\n", #UNIT "." #NAME, func_config->UNIT.NAME);

#define FOF_HEX "0x%llx"
#define FOF_DEC "%" PRIu32
#define FOF_DECU64 "%" PRIu64

#include "func_config.def"
#undef DEF_UNIT_START
#undef DEF_UNIT_END
#undef DEF_OPTION
#undef DEF_UNIT_OPTION
#undef DEF_OPTION_STR
#undef DEF_UNIT_OPTION_STR

    return 0;
}

static const char* programname;

void func_model_print_debug_masks(FILE* out)
{
    fprintf(out, "\t  List of components:\n");
#define DEBUG_MODE(string, value) fprintf(out, "\t\t" #string "\n");
#include "debug_modes.def"
#undef DEBUG_MODE
}

int func_model_print_help(FILE* out)
{
    fprintf(out, "TOSA Reference Model help\n\n");

    fprintf(out,
            "Usage: %s [-c] [-C <name=value>] [-d <Debug Mask>] [-h] [-l <verbosity>] [-F "
            "<flatconfig>]\n",
            programname);
    fprintf(out, "\t-c - Print list of config options\n");
    fprintf(out, "\t-C <name=value> - modify config option <name> to <value>\n");
    fprintf(out, "\t-d <Debug Mask - set component debug mask\n");
    func_model_print_debug_masks(out);
    fprintf(out, "\t-F <flatconfig> - parse <flatconfig> as file of config options\n");
    fprintf(out, "\t-v - Print refererence model version\n");
    fprintf(out, "\t-h - show this help message and exit\n");
    fprintf(
        out,
        "\t-i <input_tensor_name>,<filename> - set input tensor <input_tensor_name> to the values from <filename>\n");
    fprintf(out, "\t-l <verbosity> - set log verbosity\n");
    fprintf(out, "\t-o <debuglog> - set debug log file\n");
    fprintf(out, "\n");

    func_config_print_config_help(stdout);

    return 0;
}

static const char* get_arg_text(int& index, const int argc, const char** argv)
{
    if (strlen(argv[index]) > 2)
    {
        return argv[index] + 2;
    }

    if ((index + 1 == argc) || (argv[index + 1][0] == '-'))
    {
        fprintf(stderr, "No option value found for option %s\n", argv[index]);
        return "";
    }

    index++;
    return argv[index];
}

// Read the command line arguments
int func_model_parse_cmd_line(
    func_config_t* func_config, func_debug_t* func_debug, const int argc, const char** argv, const char* version)
{
    int i;
    programname = argv[0];
    for (i = 1; i < argc; i++)
    {
        // All command line arguments must begin with -X  where X is a recognized character
        if (strlen(argv[i]) < 2 || argv[i][0] != '-')
        {
            func_model_print_help(stderr);
            ARG_ERROR("Command line argument at position %d not valid: %s", i, argv[i]);
        }

        switch (argv[i][1])
        {
                // Model parameters may be overridden with the -Cname=value switch
            case 'c':
                func_config_print_config_help(stderr);
                return 1;

            case 'C':
            {
                const char *name = nullptr, *value = nullptr;

                // Break the string into name and value parts
                name  = get_arg_text(i, argc, argv);
                value = strchr(name, '=');

                if (value == nullptr)
                {
                    func_model_print_help(stderr);
                    ARG_ERROR("Cannot parse -C argument at position %d: %s", i, argv[i]);
                }

                *const_cast<char*>(value) = 0;

                if (func_model_config_set_option(func_config, name, value + 1))
                {
                    func_model_print_help(stderr);
                    ARG_ERROR("Cannot parse -C argument at position %d: %s", i, argv[i]);
                }
                break;
            }

            case 'd':
            case 'D':
            {
                func_debug_set_mask(func_debug, get_arg_text(i, argc, argv));
                break;
            }
            case 'F':
            {
                // Read a flat configuration file
                if (func_model_parse_flat_config_file(func_config, get_arg_text(i, argc, argv)))
                    return 1;

                break;
            }
            case 'h':
                func_model_print_help(stderr);
                return 1;

            case 'i':
            {
                // shortcut for '-Cinput_tensor='
                if (func_model_config_set_option(func_config, "input_tensor", get_arg_text(i, argc, argv)))
                {
                    func_model_print_help(stderr);
                    ARG_ERROR("Cannot set input tensor config value");
                }
                break;
            }
            case 'l':
            {
                // Debug verbosity/logging level
                func_debug_set_verbosity(func_debug, get_arg_text(i, argc, argv));
                break;
            }
            case 'o':
            {
                func_debug_set_file(func_debug, get_arg_text(i, argc, argv));
                break;
            }
            case 'v':
            {
                fprintf(stdout, "Model Version %s\n", version);
                return 1;
            }
            default:
                func_model_print_help(stderr);
                ARG_ERROR("Unrecognized argument at position %d: %s", i, argv[i]);
        }
    }

    return 0;
}

int func_model_parse_flat_config_file(func_config_t* func_config, const char* filename)
{
    const int MAX_LINE_LEN = 1024;

    FILE* infile = nullptr;
    char line_buf[MAX_LINE_LEN];
    int line = 1;

    infile = fopen(filename, "r");

    if (infile == nullptr)
    {
        ARG_ERROR("Cannot open config file: %s\n", filename);
    }

    while (fgets(line_buf, MAX_LINE_LEN - 1, infile) != nullptr)
    {
        char *name = line_buf, *value = nullptr, *comment = nullptr, *ptr = nullptr;

        // Remove comments
        comment = strchr(line_buf, '#');

        if (comment)
            *comment = 0;

        // Break the string into name and value parts
        name = line_buf;

        // Remove leading whitespace
        while (*name && isspace(*name))
            name++;

        // Empty line?
        if (*name == 0)
        {
            line++;
            continue;
        }

        value = strchr(name, '=');

        // Missing value
        if (value == nullptr)
        {
            ARG_ERROR("Cannot parse parameter in %s at line %d: %s", filename, line, line_buf);
        }

        // Remove the =
        *value = 0;
        value++;

        // Trim off any whitespace at the end of the value
        ptr = value;
        while (*ptr != 0 && !isspace(*ptr))
            ptr++;
        *ptr = 0;

        // Include a nested file
        if (!strcmp(name, "include"))
        {
            if (func_model_parse_flat_config_file(func_config, value))
                return 1;
            line++;
            continue;
        }

        if (func_model_config_set_option(func_config, name, value))
        {
            func_model_print_help(stderr);
            ARG_ERROR("Cannot set parameter in %s at line %d: %s", filename, line, line_buf)
        }

        line++;
    }

    fclose(infile);

    return 0;
}

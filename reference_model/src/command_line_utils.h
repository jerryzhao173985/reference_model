
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

#ifndef COMMAND_LINE_UTILS_H_
#define COMMAND_LINE_UTILS_H_

#include "func_config.h"
#include "func_debug.h"

#include <stdint.h>
#include <cxxopts.hpp>

// Read the command line arguments
int func_model_parse_cmd_line(
    func_config_t& func_config, func_debug_t& func_debug, int argc, char** argv, const char* version)
{
    try
    {
        cxxopts::Options options("tosa_reference_model", "The TOSA reference model");

        // clang-format off
        options.add_options()
        ("operator_fbs", "Flat buffer schema file", cxxopts::value<std::string>(func_config.operator_fbs), "<schema>")
        ("test_desc", "Json test descriptor", cxxopts::value<std::string>(func_config.test_desc), "<descriptor>")
        ("flatbuffer_dir", "Flatbuffer directory to load. If not specified, it will be overwritten by dirname(test_desc)",
            cxxopts::value<std::string>(func_config.flatbuffer_dir))
        ("output_dir", "Output directory to write. If not specified, it will be overwritten by dirname(test_desc)",
            cxxopts::value<std::string>(func_config.output_dir))
        ("tosa_file", "Flatbuffer file. Support .json or .tosa. Specifying this will overwrite the one initialized by --test_desc.",
            cxxopts::value<std::string>(func_config.tosa_file))
        ("ifm_name", "Input tensor name. Comma(,) separated. Specifying this will overwrite the one initialized by --test_desc.",
            cxxopts::value<std::string>(func_config.ifm_name))
        ("ifm_file", "Input tensor numpy Comma(,) separated. file to initialize with placeholder. Specifying this will overwrite the one initialized by --test_desc.",
            cxxopts::value<std::string>(func_config.ifm_file))
        ("ofm_name", "Output tensor name. Comma(,) seperated. Specifying this will overwrite the one initialized by --test_desc.",
            cxxopts::value<std::string>(func_config.ofm_name))
        ("ofm_file", "Output tensor numpy file to be generated. Comma(,) seperated. Specifying this will overwrite the one initialized by --test_desc.",
            cxxopts::value<std::string>(func_config.ofm_file))
        ("eval", "Evaluate the network (0/1)", cxxopts::value<uint32_t>(func_config.eval))
        ("fp_format", "Floating-point number dump format string (printf-style format, e.g. 0.5)",
            cxxopts::value<std::string>(func_config.fp_format))
        ("validate_only", "Validate the network, but do not read inputs or evaluate (0/1)",
            cxxopts::value<uint32_t>(func_config.validate_only))
        ("output_tensors", "Output tensors to a file (0/1)", cxxopts::value<uint32_t>(func_config.output_tensors))
        ("tosa_profile", "Set TOSA profile (0 = Base Inference, 1 = Main Inference, 2 = Main Training)",
            cxxopts::value<uint32_t>(func_config.tosa_profile))
        ("dump_intermediates", "Dump intermediate tensors (0/1)", cxxopts::value<uint32_t>(func_config.dump_intermediates))
        ("v,version", "print model version")
        ("i,input_tensor_file", "specify input tensor files", cxxopts::value<std::vector<std::string>>())
        ("l,loglevel", func_debug.get_debug_verbosity_help_string(), cxxopts::value<std::string>())
        ("o,logfile", "output log file", cxxopts::value<std::string>())
        ("d,debugmask", func_debug.get_debug_mask_help_string(), cxxopts::value<std::vector<std::string>>())
        ("h,help", "print help");
        // clang-format on

        auto result = options.parse(argc, argv);
        if (result.count("help")) {
            std::cout << options.help() << std::endl;
            return 1;
        }
        if (result.count("debugmask")) {
            auto& v = result["debugmask"].as<std::vector<std::string>>();
            for (const std::string& s : v)
                func_debug.set_mask(s);
        }
        if (result.count("loglevel")) {
            const std::string& levelstr = result["loglevel"].as<std::string>();
            func_debug.set_verbosity(levelstr);
        }
        if (result.count("logfile")) {
            func_debug.set_file(result["logfile"].as<std::string>());
        }
        if (result.count("input_tensor_file")) {
            func_config.ifm_name = result["input_tensor_file"].as<std::string>();
        }
        if (result.count("version")) {
            std::cout << "Model version " << version << std::endl;
        }
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }

    return 0;
}

#endif

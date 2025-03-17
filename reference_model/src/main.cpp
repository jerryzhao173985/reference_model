
// Copyright (c) 2020-2025, ARM Limited.
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

#include "model_runner.h"
#include "version.h"

#include "arith_util.h"
#include "command_line_utils.h"
#include "custom_op_interface.h"
#include "custom_registry.h"
#include "load_library.h"
#include "ops/op_factory.h"
#include "subgraph_traverser.h"
#include "tosa_serialization_handler.h"

#include <Eigen/CXX11/Tensor>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdio.h>

#if defined(__linux__) || defined(__APPLE__)
#include <dlfcn.h>
#define LIBTYPE void*
#define OPENLIB(libname) dlopen((libname), RTLD_LAZY)
#define LIBFUNC(lib, fn) dlsym((lib), (fn))
#define CLOSELIB(lib) dlclose((lib))
#elif _WIN32
#define NOMINMAX
#include <windows.h>
#define LIBTYPE HINSTANCE
#define OPENLIB(libname) load_library_w(libname)
#define LIBFUNC(lib, fn) GetProcAddress((lib), (fn))
#define CLOSELIB(lib) FreeLibrary((lib))

#endif

#ifdef _WIN32
#include <ctype.h>
#define strncasecmp _strnicmp
#endif

using namespace TosaReference;
using namespace tosa;
using json = nlohmann::json;

int initTestDesc(json& test_desc);

int readInputTensors(SubgraphTraverser& gt, json& test_desc);
int writeFinalTensors(SubgraphTraverser& gt, json& test_desc, const std::string& filename_prefix);
int readVariableTensors(SubgraphTraverser& gt, json test_desc);
int writeVariableTensors(SubgraphTraverser& gt, json test_desc);
int loadSharedLibs(std::string& custom_op_lib_path);
int loadGraph(TosaSerializationHandler& tsh, json& test_desc);
void parse_value(const std::string& text, tosa_level_t& value);
const std::string getResultFilenamePrefix();
bool isComplianceBoundsModeNeeded(json& test_desc);

int main(int argc, char** argv)
{
    TosaVersion model_version(TOSA_REFERENCE_MODEL_VERSION_MAJOR, TOSA_REFERENCE_MODEL_VERSION_MINOR,
                              TOSA_REFERENCE_MODEL_VERSION_PATCH, TOSA_REFERENCE_MODEL_VERSION_DRAFT);

    // Initialize configuration and debug subsystems
    g_func_debug.init_debug(0);

    if (func_model_parse_cmd_line(g_func_config, g_func_debug, argc, argv, model_version.to_string().c_str()))
    {
        return 1;
    }

    TosaSerializationHandler tsh;
    TosaVersion::compat_t is_compat = TosaVersion::is_compatible(model_version, tsh.GetVersion());

    switch (is_compat)
    {
        case TosaVersion::compat_t::COMPLETELY_COMPATIBLE:
            break;
        case TosaVersion::compat_t::BACKWARD_COMPATIBLE:
            printf("WARNING: Reference model version %s is backward compatible with serializer version %s\n",
                   model_version.to_string().c_str(), tsh.GetVersion().to_string().c_str());
            break;
        case TosaVersion::compat_t::NOT_COMPATIBLE:
            printf("ERROR: Reference model version %s is not compatible with serializer version %s\n",
                   model_version.to_string().c_str(), tsh.GetVersion().to_string().c_str());
            return TOSA_VERSION_MISMATCH;
    }

    json test_desc;

    // Initialize test descriptor
    if (initTestDesc(test_desc))
    {
        FATAL_ERROR("Unable to load test json");
    }

    // load shared libs if specified
    if (g_func_config.custom_op_lib_path != "")
    {
        if (loadSharedLibs(g_func_config.custom_op_lib_path))
        {
            FATAL_ERROR("Shared library specified but not loaded successfully");
        }
    }

    if (loadGraph(tsh, test_desc))
    {
        FATAL_ERROR("Unable to load graph");
    }

    GraphStatus status = GraphStatus::TOSA_VALID;

    if (isComplianceBoundsModeNeeded(test_desc))
    {
        if (g_func_config.precise_mode)
        {
            // Precise mode enabled and we have a compliance test
            g_func_config.compliance_mode = true;
        }
        else
        {
            // Warn about precise mode for compliance tests
            DEBUG_INFO(CONFIG, "Compliance test: NOTE - enable precise mode for compliance results")
        }
    }

    // max of 2 runs, second run only happens when precise_mode is set, to do a bounds_mode run
    for (int run = 0; run < 2; run++)
    {
        SubgraphTraverser main_gt(tsh.GetMainRegion()->GetBlockByName("main"), &tsh, nullptr);

        if (main_gt.initializeGraph())
        {
            WARNING("Unable to initialize main graph traverser.");
            goto done;
        }

        if (main_gt.linkTensorsAndNodes())
        {
            WARNING("Failed to link tensors and nodes");
            goto done;
        }

        if (main_gt.validateGraph())
        {
            WARNING("Failed to validate graph. Evaluation aborted.");
            goto done;
        }

        if (main_gt.allocateInputTensors())
        {
            WARNING("Failed to allocate input tensors. Evaluation aborted.");
            goto done;
        }

        if (g_func_config.validate_only)
        {
            goto done;
        }

        if (readInputTensors(main_gt, test_desc))
        {
            FATAL_ERROR("Unable to read input tensors");
        }

        if (!g_func_config.eval)
        {
            goto done;
        }

        if (g_func_config.initialize_variable_tensor_from_numpy)
        {
            if (readVariableTensors(main_gt, test_desc))
            {
                FATAL_ERROR("Unable to read variable tensors");
            }
        }

        // evaluateAll() returns 1 if graph evaluation is forced to be terminated earlier.
        if (main_gt.evaluateAll())
        {
            ASSERT_MSG(main_gt.getGraphStatus() != GraphStatus::TOSA_VALID,
                       "Upon evaluateAll() returning 1, graph can not be VALID.");
        }
        else
        {
            ASSERT_MSG(main_gt.getGraphStatus() == GraphStatus::TOSA_VALID ||
                           main_gt.getGraphStatus() == GraphStatus::TOSA_UNPREDICTABLE,
                       "Upon evaluateAll() returning 0, graph can only be VALID/UNPREDICTABLE.");
        }

        // Only generate output tensor if graph is valid.
        if (main_gt.getGraphStatus() == GraphStatus::TOSA_VALID)
        {
            // make sure output tensor is evaluated and show its value
            int num_output_tensors = main_gt.getNumOutputTensors();
            bool all_output_valid  = true;
            for (int i = 0; i < num_output_tensors; i++)
            {
                const Tensor* ct = main_gt.getOutputTensor(i);
                ASSERT_MEM(ct);
                if (!ct->getIsValid())
                {
                    ct->dumpTensorParams(g_func_debug.func_debug_file);
                    if (DEBUG_ENABLED(DEBUG_VERB_HIGH, GT))
                    {
                        ct->dumpTensor(g_func_debug.func_debug_file);
                    }
                    all_output_valid = false;
                }
            }
            if (!all_output_valid)
            {
                main_gt.dumpGraph(g_func_debug.func_debug_file);
                FATAL_ERROR(
                    "SubgraphTraverser \"main\" error: Output tensors are not all valid at the end of evaluation.");
            }

            if (g_func_config.output_tensors)
            {
                if (writeFinalTensors(main_gt, test_desc, getResultFilenamePrefix()))
                {
                    WARNING("Errors encountered in saving output tensors");
                }

                if (writeVariableTensors(main_gt, test_desc))
                {
                    WARNING("Errors encountered in writing variable tensors");
                }
            }
        }

    done:
        status = main_gt.getGraphStatus();
        switch (status)
        {
            case GraphStatus::TOSA_VALID:
                // Result is valid.
                break;
            case GraphStatus::TOSA_UNPREDICTABLE:
                fprintf(stderr, "Graph result: UNPREDICTABLE.\n");
                break;
            case GraphStatus::TOSA_ERROR:
                fprintf(stderr, "Graph result: ERROR.\n");
                break;
            default:
                fprintf(stderr, "Unknown graph status code=%d.\n", (int)main_gt.getGraphStatus());
        }

        if (run == 0 && status == GraphStatus::TOSA_VALID && g_func_config.compliance_mode && g_func_config.eval)
        {
            // when first run result is valid and precise mode and eval is true: turn on bounds_mode for second run
            DEBUG_INFO(CONFIG, "Compliance test: Evaluating the graph again to produce bounds results")
            g_func_config.bounds_mode = true;
            continue;
        }

        // otherwise, do only one run
        break;
    }

    g_func_debug.fini_debug();
    return (int)status;
}

int loadSharedLibs(std::string& custom_op_lib_path)
{
    // Load the shared_lib
    const char* path   = custom_op_lib_path.c_str();
    LIBTYPE lib_handle = OPENLIB(path);
    if (lib_handle == nullptr)
    {
        FATAL_ERROR("Library %s does not exist\n", custom_op_lib_path.c_str());
    }

    typedef int (*get_customOp_function_t)(registration_callback_t registration_func);
    auto get_customOp_creation_funcs =
        reinterpret_cast<get_customOp_function_t>(LIBFUNC(lib_handle, "getCustomOpCreationFuncs"));
    if (get_customOp_creation_funcs == nullptr)
    {
        FATAL_ERROR("Can't find the getCustomOpCreationFuncs \n");
    }

    return get_customOp_creation_funcs(&MasterRegistry::register_function);
}

int loadGraph(TosaSerializationHandler& tsh, json& test_desc)
{
    const std::string error_msg1 = "Check \"tosa_file\" in .json specified by --tosa_desc";
    const std::string error_msg2 = " or via arguments --tosa_file & --flatbuffer_dir";

    if (test_desc["tosa_file"].get<std::string>().size() <= 0)
    {
        FATAL_ERROR("Missing tosa_file.\n%s", error_msg1.c_str());
    }

    std::string graph_fullname_str = g_func_config.flatbuffer_dir + "/" + test_desc["tosa_file"].get<std::string>();

    const char JSON_EXT[] = ".json";
    int is_json           = 0;
    {
        // look for JSON file extension
        size_t suffix_len = sizeof(JSON_EXT) - 1;
        size_t str_len    = graph_fullname_str.size();

        if (str_len > suffix_len &&
            strncasecmp(graph_fullname_str.c_str() + (str_len - suffix_len), JSON_EXT, suffix_len) == 0)
        {
            is_json = 1;
        }
    }

    if (is_json)
    {
        if (tsh.LoadFileSchema(g_func_config.operator_fbs.c_str()))
        {
            FATAL_ERROR("\nJSON file detected.  Unable to load TOSA flatbuffer schema from: %s\nCheck --operator_fbs "
                        "is set correctly",
                        g_func_config.operator_fbs.c_str());
        }

        if (tsh.LoadFileJson(graph_fullname_str.c_str()))
        {
            FATAL_ERROR("\nError loading JSON graph file: %s\n%s%s\nCheck --operator_fbs is using correct version",
                        graph_fullname_str.c_str(), error_msg1.c_str(), error_msg2.c_str());
        }
    }
    else
    {
        if (tsh.LoadFileTosaFlatbuffer(graph_fullname_str.c_str()))
        {
            FATAL_ERROR("\nError loading TOSA flatbuffer file: %s\n%s%s", graph_fullname_str.c_str(),
                        error_msg1.c_str(), error_msg2.c_str());
        }
    }

    return 0;
}

int readInputTensors(SubgraphTraverser& gt, json& test_desc)
{
    int tensorCount = gt.getNumInputTensors();
    Tensor* tensor;

    try
    {
        if ((tensorCount != (int)test_desc["ifm_name"].size()) || (tensorCount != (int)test_desc["ifm_file"].size()))
        {
            WARNING("Number of input tensors(%d) doesn't match name(%ld)/file(%ld) in test descriptor.", tensorCount,
                    test_desc["ifm_name"].size(), test_desc["ifm_file"].size());
            return 1;
        }

        for (int i = 0; i < tensorCount; i++)
        {
            tensor = gt.getInputTensorByName(test_desc["ifm_name"][i].get<std::string>());
            if (!tensor)
            {
                WARNING("Unable to find input tensor %s", test_desc["ifm_name"][i].get<std::string>().c_str());
                return 1;
            }

            std::string filename_str = g_func_config.flatbuffer_dir + "/" + test_desc["ifm_file"][i].get<std::string>();

            DEBUG_MED(GT, "Loading input tensor %s from filename: %s", tensor->getName().c_str(), filename_str.c_str());

            if (!tensor->is_allocated())
            {
                WARNING("Tensor %s is not allocated before being initialized", tensor->getName().c_str());
                return 1;
            }

            if (tensor->readFromNpyFile(filename_str.c_str()))
            {
                WARNING("Unable to read input tensor %s from filename: %s", tensor->getName().c_str(),
                        filename_str.c_str());
                tensor->dumpTensorParams(g_func_debug.func_debug_file);
                return 1;
            }

            // Push ready consumers to the next node list
            for (auto gn : tensor->getConsumers())
            {
                if (gn->hasAllInputsReady() && !gn->getOnNextNodeList() && !gn->getEvaluated())
                {
                    gt.addToNextNodeList(gn);
                }
            }
        }
    }
    catch (nlohmann::json::type_error& e)
    {
        WARNING("Fail accessing test descriptor: %s", e.what());
        return 1;
    }

    if (DEBUG_ENABLED(DEBUG_VERB_HIGH, GT))
    {
        gt.dumpNextNodeList(g_func_debug.func_debug_file);
    }

    return 0;
}

const std::string getResultFilenamePrefix()
{
    return g_func_config.bounds_mode ? "bounds_" : "";
}

// returns true if test_desc contains a "meta" object containing a "compliance"
// object which contains "tensors" and one of those has a "mode" whose value is
// "DOT_PRODUCT" or "ABS_ERROR" or "FP_SPECIAL" or "RESCALE_INEXACT"
bool isComplianceBoundsModeNeeded(json& test_desc)
{
    if (test_desc.contains("meta") && test_desc["meta"].contains("compliance") &&
        test_desc["meta"]["compliance"].contains("tensors"))
    {
        for (auto t : test_desc["meta"]["compliance"]["tensors"])
        {
            if (t.contains("mode") && (t["mode"] == "DOT_PRODUCT" || t["mode"] == "ABS_ERROR" ||
                                       t["mode"] == "FP_SPECIAL" || t["mode"] == "RESCALE_INEXACT"))
            {
                return true;
            }
        }
    }
    return false;
}

int writeFinalTensors(SubgraphTraverser& gt, json& test_desc, const std::string& filename_prefix)
{
    int tensorCount = gt.getNumOutputTensors();
    const Tensor* tensor;

    try
    {
        if ((tensorCount != (int)test_desc["ofm_name"].size()) || (tensorCount != (int)test_desc["ofm_file"].size()))
        {
            WARNING("Number of output tensors(%d) doesn't match name(%ld)/file(%ld) in test descriptor.", tensorCount,
                    test_desc["ofm_name"].size(), test_desc["ofm_file"].size());
            return 1;
        }

        for (int i = 0; i < tensorCount; i++)
        {
            tensor = gt.getOutputTensorByName(test_desc["ofm_name"][i].get<std::string>());
            if (!tensor)
            {
                WARNING("Unable to find output tensor %s", test_desc["ofm_name"][i].get<std::string>().c_str());
                return 1;
            }

            std::string filename_str =
                g_func_config.output_dir + "/" + filename_prefix + test_desc["ofm_file"][i].get<std::string>();

            DEBUG_MED(GT, "Writing output tensor[%d] %s to filename: %s", i, tensor->getName().c_str(),
                      filename_str.c_str());

            if (tensor->writeToNpyFile(filename_str.c_str()))
            {
                WARNING("Unable to write output tensor[%d] %s to filename: %s", i, tensor->getName().c_str(),
                        filename_str.c_str());
                return 1;
            }
        }
    }
    catch (nlohmann::json::type_error& e)
    {
        WARNING("Fail accessing test descriptor: %s", e.what());
        return 1;
    }

    return 0;
}

int readVariableTensors(SubgraphTraverser& gt, json test_desc)
{
    int tensorCount = gt.getNumVariableTensors();
    Tensor* tensor;

    try
    {
        if ((tensorCount != (int)test_desc["variable_name"].size()) ||
            (tensorCount != (int)test_desc["variable_file"].size()))
        {
            WARNING("Number of variable tensors(%d) doesn't match name(%ld)/file(%ld)in test descriptor.", tensorCount,
                    test_desc["variable_name"].size(), test_desc["variable_file"].size());
            return 1;
        }

        for (int i = 0; i < tensorCount; i++)
        {
            tensor = gt.getVariableTensorByName(test_desc["variable_name"][i].get<std::string>());
            if (!tensor)
            {
                WARNING("Unable to find variable tensor %s", test_desc["variable_name"][i].get<std::string>().c_str());
                return 1;
            }

            std::string filename_str =
                g_func_config.flatbuffer_dir + "/" + test_desc["variable_file"][i].get<std::string>();

            DEBUG_MED(GT, "Loading variable tensor %s from filename: %s", tensor->getName().c_str(),
                      filename_str.c_str());

            if (!tensor->is_allocated())
            {
                WARNING("Tensor %s is not allocated before being initialized", tensor->getName().c_str());
                return 1;
            }

            if (tensor->readFromNpyFile(filename_str.c_str()))
            {
                WARNING("Unable to read variable tensor %s from filename: %s", tensor->getName().c_str(),
                        filename_str.c_str());
                tensor->dumpTensorParams(g_func_debug.func_debug_file);
                return 1;
            }

            // Push ready consumers to the next node list
            for (auto gn : tensor->getConsumers())
            {
                if (gn->hasAllInputsReady() && !gn->getOnNextNodeList() && !gn->getEvaluated())
                {
                    gt.addToNextNodeList(gn);
                }
            }
        }
    }
    catch (nlohmann::json::type_error& e)
    {
        WARNING("Fail accessing test descriptor: %s", e.what());
        return 1;
    }

    if (DEBUG_ENABLED(DEBUG_VERB_HIGH, GT))
    {
        gt.dumpNextNodeList(g_func_debug.func_debug_file);
    }

    return 0;
}

int writeVariableTensors(SubgraphTraverser& gt, json test_desc)
{
    int tensorCount = gt.getNumVariableTensors();
    const Tensor* tensor;

    try
    {
        if ((tensorCount != (int)test_desc["variable_name"].size()) ||
            (tensorCount != (int)test_desc["variable_file"].size()))
        {
            WARNING("Number of variable tensors(%d) doesn't match name(%ld)/file(%ld) in test descriptor.", tensorCount,
                    test_desc["variable_name"].size(), test_desc["variable_file"].size());
            return 1;
        }

        for (int i = 0; i < tensorCount; i++)
        {
            tensor = gt.getVariableTensorByName(test_desc["variable_name"][i].get<std::string>());
            if (!tensor)
            {
                WARNING("Unable to find variable tensor %s", test_desc["variable_name"][i].get<std::string>().c_str());
                return 1;
            }

            std::string filename_str =
                g_func_config.output_dir + "/" + test_desc["variable_file"][i].get<std::string>();

            DEBUG_MED(GT, "Writing variable tensor[%d] %s to filename: %s", i, tensor->getName().c_str(),
                      filename_str.c_str());
            if (!tensor->is_allocated())
            {
                WARNING("Tensor %s is no longer allocated", tensor->getName().c_str());
                return 1;
            }
            if (tensor->writeToNpyFile(filename_str.c_str()))
            {
                WARNING("Unable to write variable tensor[%d] %s to filename: %s", i, tensor->getName().c_str(),
                        filename_str.c_str());
                return 1;
            }
        }
    }
    catch (nlohmann::json::type_error& e)
    {
        WARNING("Fail accessing test descriptor: %s", e.what());
        return 1;
    }

    return 0;
}

// Read "foo,bar,..." and return std::vector({foo, bar, ...})
std::vector<std::string> parseFromString(std::string raw_str)
{
    bool last_pair               = false;
    std::string::size_type start = 0, end;
    std::string name;

    std::vector<std::string> result;
    do
    {
        end = raw_str.find(',', start);
        if (end == std::string::npos)
            last_pair = true;

        // The second parameter holds for number of characters to include in the substring,
        // not for the index of the end of the capture.
        name = raw_str.substr(start, end - start);

        result.push_back(name);

        start = end + 1;    // skip comma
    } while (!last_pair);

    return result;
}

int initTestDesc(json& test_desc)
{
    std::ifstream ifs(g_func_config.test_desc);

    if (ifs.good())
    {
        try
        {
            test_desc = nlohmann::json::parse(ifs);
        }
        catch (nlohmann::json::parse_error& e)
        {
            WARNING("Error parsing test descriptor json: %s", e.what());
            return 1;
        }
    }
    else
    {
        // Users can also specify description info using command line arguments
        // If they miss any info from command line AND no test_desc is provided,
        // return error code
        if (g_func_config.tosa_file.empty() || g_func_config.ifm_name.empty() || g_func_config.ifm_file.empty() ||
            g_func_config.ofm_name.empty() || g_func_config.ofm_file.empty())
        {
            WARNING("Cannot open input file: %s", g_func_config.test_desc.c_str());
            return 1;
        }
    }

    // Overwrite flatbuffer_dir/output_dir with dirname(g_func_config.test_desc) if it's not specified.
    if (g_func_config.flatbuffer_dir.empty() || g_func_config.output_dir.empty())
    {
        auto slash_pos = g_func_config.test_desc.find_last_of("/\\");
        std::string test_dir;
        if (slash_pos != std::string::npos)
        {
            test_dir = g_func_config.test_desc.substr(0, slash_pos);
        }
        else
        {
            test_dir = std::string(".");
        }
        if (g_func_config.flatbuffer_dir.empty())
        {
            g_func_config.flatbuffer_dir = test_dir;
        }
        if (g_func_config.output_dir.empty())
        {
            g_func_config.output_dir = test_dir;
        }
    }

    // Overwrite test_desc["tosa_file"] if --tosa_file specified.
    if (!g_func_config.tosa_file.empty())
    {
        test_desc["tosa_file"] = g_func_config.tosa_file;
    }

    // Overwrite test_desc["ifm_name"] if --ifm_name specified.
    if (!g_func_config.ifm_name.empty())
    {
        std::vector<std::string> ifm_name_vec = parseFromString(g_func_config.ifm_name);
        test_desc["ifm_name"]                 = ifm_name_vec;
    }

    // Overwrite test_desc["ifm_file"] if --ifm_file specified.
    if (!g_func_config.ifm_file.empty())
    {
        std::vector<std::string> ifm_file_vec = parseFromString(g_func_config.ifm_file);
        test_desc["ifm_file"]                 = ifm_file_vec;
    }

    // Overwrite test_desc["ofm_name"] if --ofm_name specified.
    if (!g_func_config.ofm_name.empty())
    {
        std::vector<std::string> ofm_name_vec = parseFromString(g_func_config.ofm_name);
        test_desc["ofm_name"]                 = ofm_name_vec;
    }

    // Overwrite test_desc["ofm_file"] if --ofm_file specified.
    if (!g_func_config.ofm_file.empty())
    {
        std::vector<std::string> ofm_file_vec = parseFromString(g_func_config.ofm_file);
        test_desc["ofm_file"]                 = ofm_file_vec;
    }

    // Overwrite test_desc["variable_name"] if --variable_name= specified.
    std::string variable_name_str(g_func_config.variable_name);
    if (!variable_name_str.empty())
    {
        std::vector<std::string> variable_name_vec = parseFromString(variable_name_str);
        test_desc["variable_name"]                 = variable_name_vec;
    }

    // Overwrite test_desc["variable_file"] if --variable_file= specified.
    std::string variable_file_str(g_func_config.variable_file);
    if (!variable_file_str.empty())
    {
        std::vector<std::string> variable_file_vec = parseFromString(variable_file_str);
        test_desc["variable_file"]                 = variable_file_vec;
    }

    // Overwirte --terminate_early command-line option, if test_desc["terminate_early"] specified.
    if (test_desc.contains("terminate_early"))
    {
        if (!(test_desc["terminate_early"].is_boolean()))
        {
            WARNING("terminate_early is not a boolean in the JSON file.");
            return 1;
        }
        g_func_config.terminate_early = test_desc["terminate_early"].get<bool>();
    }

    return 0;
}

void parse_value(const std::string& text, tosa_level_t& value)
{

    if (text == "NONE")
        value = func_config_t::NONE;
    else if (text == "EIGHTK")
        value = func_config_t::EIGHTK;
    else
        throw cxxopts::argument_incorrect_type("TOSA_LEVEL");
    return;
}
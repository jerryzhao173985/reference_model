
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

#include "model_runner.h"
#include "version.h"

#include "command_line_utils.h"
#include "ops/op_factory.h"
#include "subgraph_traverser.h"
#include "tosa_serialization_handler.h"
#include "arith_util.h"

#include <fstream>
#include <iostream>
#include <stdio.h>
#include <Eigen/CXX11/Tensor>
#include <nlohmann/json.hpp>

using namespace TosaReference;
using namespace tosa;
using json = nlohmann::json;

int initTestDesc(json& test_desc);
int readInputTensors(SubgraphTraverser& gt, json test_desc);
int writeFinalTensors(SubgraphTraverser& gt, json test_desc);
int loadGraph(TosaSerializationHandler& tsh, json test_desc);

int main(int argc, char** argv)
{
    TosaVersion model_version(TOSA_REFERENCE_MODEL_VERSION_MAJOR,
                              TOSA_REFERENCE_MODEL_VERSION_MINOR,
                              TOSA_REFERENCE_MODEL_VERSION_PATCH,
                              TOSA_REFERENCE_MODEL_VERSION_DRAFT);

    // Initialize configuration and debug subsystems
    g_func_debug.init_debug(0);

    if (func_model_parse_cmd_line(g_func_config, g_func_debug, argc, argv, model_version.to_string().c_str()))
    {
        return 1;
    }

    TosaSerializationHandler tsh;
    TosaVersion::compat_t is_compat = model_version.is_compatible(tsh.GetVersion());
    switch (is_compat)
    {
        case TosaVersion::compat_t::COMPLETELY_COMPATIBLE:
            break;
        case TosaVersion::compat_t::PARTIALLY_COMPATIBLE:
            printf("WARNING: Reference model version %s is partially compatible with serializer version %s\n",
                   model_version.to_string().c_str(), tsh.GetVersion().to_string().c_str());
            break;
        case TosaVersion::compat_t::NOT_COMPATIBLE:
            printf("ERROR: Reference model version %s is not compatible with serializer version %s\n",
                   model_version.to_string().c_str(), tsh.GetVersion().to_string().c_str());
            return TOSA_VERSION_MISMATCH;
    }

    g_func_config.float_is_big_endian = float_is_big_endian();

    json test_desc;

    // Initialize test descriptor
    if (initTestDesc(test_desc))
    {
        FATAL_ERROR("Unable to load test json");
    }

    if (loadGraph(tsh, test_desc))
    {
        FATAL_ERROR("Unable to load graph");
    }

    SubgraphTraverser main_gt(tsh.GetMainRegion()->GetBlocks().front(), &tsh, nullptr);

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

    if (main_gt.allocateTensor())
    {
        WARNING("Failed to allocate tensor. Evaluation aborted.");
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

    if (g_func_config.eval)
    {

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
                if (writeFinalTensors(main_gt, test_desc))
                {
                    WARNING("Errors encountered in saving output tensors");
                }
            }
        }
    }

done:
    switch (main_gt.getGraphStatus())
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

    g_func_debug.fini_debug();

    return (int)main_gt.getGraphStatus();
}

int loadGraph(TosaSerializationHandler& tsh, json test_desc)
{
    char graph_fullname[1024];
    const std::string error_msg1 = "Check \"tosa_file\" in .json specified by --tosa_desc";
    const std::string error_msg2 = " or via arguments --tosa_file & --flatbuffer_dir";

    if (strlen(test_desc["tosa_file"].get<std::string>().c_str()) <= 0)
    {
        FATAL_ERROR("Missing tosa_file.\n%s", error_msg1.c_str());
    }

    snprintf(graph_fullname, sizeof(graph_fullname), "%s/%s", g_func_config.flatbuffer_dir.c_str(),
             test_desc["tosa_file"].get<std::string>().c_str());

    const char JSON_EXT[] = ".json";
    int is_json           = 0;
    {
        // look for JSON file extension
        size_t suffix_len = strlen(JSON_EXT);
        size_t str_len    = strlen(graph_fullname);

        if (str_len > suffix_len && strncasecmp(graph_fullname + (str_len - suffix_len), JSON_EXT, suffix_len) == 0)
        {
            is_json = 1;
        }
    }

    if (is_json)
    {
        if (tsh.LoadFileSchema(g_func_config.operator_fbs.c_str()))
        {
            FATAL_ERROR("\nJSON file detected.  Unable to load TOSA flatbuffer schema from: %s\nCheck --operator_fbs is set correctly",
                        g_func_config.operator_fbs.c_str());
        }

        if (tsh.LoadFileJson(graph_fullname))
        {
            FATAL_ERROR("\nError loading JSON graph file: %s\n%s%s\nCheck --operator_fbs is using correct version",
                        graph_fullname, error_msg1.c_str(), error_msg2.c_str());
        }
    }
    else
    {
        if (tsh.LoadFileTosaFlatbuffer(graph_fullname))
        {
            FATAL_ERROR(
                "\nError loading TOSA flatbuffer file: %s\n%s%s",
                graph_fullname, error_msg1.c_str(), error_msg2.c_str());
        }
    }

    return 0;
}

int readInputTensors(SubgraphTraverser& gt, json test_desc)
{
    int tensorCount = gt.getNumInputTensors();
    Tensor* tensor;
    char filename[1024];

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

            snprintf(filename, sizeof(filename), "%s/%s", g_func_config.flatbuffer_dir.c_str(),
                     test_desc["ifm_file"][i].get<std::string>().c_str());

            DEBUG_MED(GT, "Loading input tensor %s from filename: %s", tensor->getName().c_str(), filename);

            if (!tensor->is_allocated())
            {
                WARNING("Tensor %s is not allocated before being initialized", tensor->getName().c_str());
                return 1;
            }

            if (tensor->readFromNpyFile(filename))
            {
                WARNING("Unable to read input tensor %s from filename: %s", tensor->getName().c_str(), filename);
                tensor->dumpTensorParams(g_func_debug.func_debug_file);
                return 1;
            }

            // Push ready consumers to the next node list
            for (auto gn : tensor->getConsumers())
            {
                if (gn->hasAllInputsReady() && !gn->getOnNextNodeList())
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

int writeFinalTensors(SubgraphTraverser& gt, json test_desc)
{
    int tensorCount = gt.getNumOutputTensors();
    const Tensor* tensor;
    char filename[1024];

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

            snprintf(filename, sizeof(filename), "%s/%s", g_func_config.output_dir.c_str(),
                     test_desc["ofm_file"][i].get<std::string>().c_str());

            DEBUG_MED(GT, "Writing output tensor[%d] %s to filename: %s", i, tensor->getName().c_str(), filename);

            if (tensor->writeToNpyFile(filename))
            {
                WARNING("Unable to write output tensor[%d] %s to filename: %s", i, tensor->getName().c_str(), filename);
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

        name = raw_str.substr(start, end);

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
        WARNING("Cannot open input file: %s", g_func_config.test_desc.c_str());
        return 1;
    }

    // Overwrite flatbuffer_dir/output_dir with dirname(g_func_config.test_desc) if it's not specified.
    if (g_func_config.flatbuffer_dir.empty() || g_func_config.output_dir.empty())
    {
        std::string test_dir = g_func_config.test_desc.substr(0, g_func_config.test_desc.find_last_of("/\\"));
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

    return 0;
}

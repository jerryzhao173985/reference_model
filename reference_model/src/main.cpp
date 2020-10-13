
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

#include <stdio.h>

#include "flatbuffers/idl.h"
#include "flatbuffers/util.h"
#include "model_common.h"
#include "ops/op_factory.h"
#include "subgraph_traverser.h"
#include "tosa_serialization_handler.h"
#include <Eigen/CXX11/Tensor>
#include <iostream>

using namespace TosaReference;
using namespace tosa;

// Global instantiation of configuration and debug objects
func_config_t g_func_config;
func_debug_t g_func_debug;

int readInputTensors(SubgraphTraverser& gt);
int writeFinalTensors(SubgraphTraverser& gt);
int loadGraph(TosaSerializationHandler& tsh);

int main(int argc, const char** argv)
{
    // Initialize configuration and debug subsystems
    func_model_init_config();
    func_model_set_default_config(&g_func_config);
    func_init_debug(&g_func_debug, 0);
    TosaSerializationHandler tsh;

    if (func_model_parse_cmd_line(&g_func_config, &g_func_debug, argc, argv))
    {
        return 1;
    }

    if (loadGraph(tsh))
    {
        SIMPLE_FATAL_ERROR("Unable to load graph");
    }

    // load json first since it's easier debugging
    SubgraphTraverser main_gt(tsh.GetMainBlock(), &tsh);

    if (main_gt.initializeGraph())
    {
        SIMPLE_FATAL_ERROR("Unable to initialize graph traverser: \"main\"");
    }

    if (main_gt.linkTensorsAndNodes())
    {
        SIMPLE_FATAL_ERROR("Failed to link tensors and nodes");
    }

    if (main_gt.validateGraph())
    {
        SIMPLE_FATAL_ERROR("Failed to validate graph");
    }

    if (g_func_config.validate_only)
    {
        goto done;
    }

    if (readInputTensors(main_gt))
    {
        SIMPLE_FATAL_ERROR("Unable to read input tensors");
    }

    if (g_func_config.eval)
    {

        if (main_gt.evaluateAll())
        {
            SIMPLE_FATAL_ERROR("Error evaluating network.  Giving up.");
        }

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
            SIMPLE_FATAL_ERROR(
                "SubgraphTraverser \"main\" error: Output tensors are not all valid at the end of evaluation.");
        }

        if (g_func_config.output_tensors)
        {
            if (writeFinalTensors(main_gt))
            {
                WARNING("Errors encountered in saving output tensors");
            }
        }
    }

done:
    func_fini_debug(&g_func_debug);
    func_model_config_cleanup();

    return 0;
}

int loadGraph(TosaSerializationHandler& tsh)
{
    char graph_fullname[1024];

    snprintf(graph_fullname, sizeof(graph_fullname), "%s/%s", g_func_config.subgraph_dir, g_func_config.subgraph_file);

    if (strlen(graph_fullname) <= 2)
    {
        func_model_print_help(stderr);
        SIMPLE_FATAL_ERROR("Missing required argument: Check -Csubgraph_file=");
    }

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
        if (tsh.LoadFileSchema(g_func_config.operator_fbs))
        {
            SIMPLE_FATAL_ERROR(
                "\nJSON file detected.  Unable to load TOSA flatbuffer schema from: %s\nCheck -Coperator_fbs=",
                g_func_config.operator_fbs);
        }

        if (tsh.LoadFileJson(graph_fullname))
        {
            SIMPLE_FATAL_ERROR("\nError loading JSON graph file: %s\nCheck -Csubgraph_file= and -Csubgraph_dir=",
                               graph_fullname);
        }
    }
    else
    {
        if (tsh.LoadFileTosaFlatbuffer(graph_fullname))
        {
            SIMPLE_FATAL_ERROR("\nError loading TOSA flatbuffer file: %s\nCheck -Csubgraph_file= and -Csubgraph_dir=",
                               graph_fullname);
        }
    }

    return 0;
}

int readInputTensors(SubgraphTraverser& gt)
{
    int tensorCount = gt.getNumInputTensors();
    Tensor* tensor;
    char filename[1024];

    // assuming filename doesn't have colons(:)
    std::map<std::string, std::string> input_tensor_map;
    std::string raw_str(g_func_config.input_tensor);
    std::string name, npy;
    bool last_pair = false;

    std::string::size_type pair_start = 0, pair_end, colons_pos;
    do
    {
        pair_end = raw_str.find(',', pair_start);
        if (pair_end == std::string::npos)
            last_pair = true;

        colons_pos = raw_str.find(':', pair_start);

        name = raw_str.substr(pair_start, colons_pos - pair_start);
        npy  = raw_str.substr(colons_pos + 1, pair_end - colons_pos - 1);

        // Empty strings can make it to here
        if (name.length() == 0 || npy.length() == 0)
            break;

        input_tensor_map[name] = npy;

        pair_start = pair_end + 1;    // skip colons
    } while (!last_pair);

    if ((size_t)tensorCount != input_tensor_map.size())
    {
        WARNING("graph has %lu input placeholders, but %lu initialized", tensorCount, input_tensor_map.size());
        return 1;
    }

    for (auto& tensor_pair : input_tensor_map)
    {
        tensor = gt.getInputTensorByName(tensor_pair.first);
        if (!tensor)
        {
            WARNING("Unable to find input tensor %s", tensor_pair.first.c_str());
            return 1;
        }

        snprintf(filename, sizeof(filename), "%s/%s", g_func_config.input_dir, tensor_pair.second.c_str());

        DEBUG_MED(GT, "Loading input tensor %s from filename: %s", tensor->getName().c_str(), filename);

        if (tensor->allocate())
        {
            WARNING("Fail to allocate tensor %s", tensor->getName().c_str());
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

    if (DEBUG_ENABLED(DEBUG_VERB_HIGH, GT))
    {
        gt.dumpNextNodeList(g_func_debug.func_debug_file);
    }

    return 0;
}

int writeFinalTensors(SubgraphTraverser& gt)
{
    int tensorCount = gt.getNumOutputTensors();
    const Tensor* tensor;
    char filename[1024];

    for (int i = 0; i < tensorCount; i++)
    {
        tensor = gt.getOutputTensor(i);
        if (!tensor)
        {
            WARNING("Unable to find output tensor[%d]", i);
            return 1;
        }

        snprintf(filename, sizeof(filename), "%s/%s%s.npy", g_func_config.output_dir,
                 g_func_config.output_tensor_prefix, tensor->getName().c_str());

        DEBUG_MED(GT, "Writing output tensor[%d] %s to filename: %s", i, tensor->getName().c_str(), filename);

        if (tensor->writeToNpyFile(filename))
        {
            WARNING("Unable to write output tensor[%d] %s to filename: %s", i, tensor->getName().c_str(), filename);
            return 1;
        }
    }

    return 0;
}

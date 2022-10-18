
// Copyright (c) 2022, ARM Limited.
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

#ifndef MODEL_RUNNER_H_
#define MODEL_RUNNER_H_

#include "model_common.h"
#include "graph_status.h"

#include "tosa_serialization_handler.h"

namespace TosaReference
{

class ModelRunnerImpl;

/*
 * This interface allows a user to initialize, run and get the output from a model.
 * See model_runner_simple_sample.cpp for example on how this interface can be used.
 */
class IModelRunner
{
public:
    IModelRunner();
    IModelRunner(const func_config_t& func_config, const func_debug_t& func_debug);

    ~IModelRunner();

    /*
     * Functional and debug configurations can also be set.
     * See func_config.h and func_debug.h for possible options.
     */
    void setFuncConfig(func_config_t& func_config);
    void setFuncDebug(func_debug_t& func_debug);

    /*
     * Initialize the model.
     * The TosaSerializationHandler is validated and then converted to a SubgraphTraverser internally.
     * This SubgraphTraverser is initialized, allocated and validated.
     * The status of the graph will be returned upon completion.
     */
    GraphStatus initialize(tosa::TosaSerializationHandler& serialization_handler);

    /*
     * Run the model using the internal SubgraphTraverser created during initialization.
     * If validate_only is specified run() will simply return the graph status.
     * Otherwise, the graph will be run and the output tensors will be generated if the graph is valid.
     * The output tensors can then be retrieved with getOutput().
     * NOTE: initialize() must be called before run(). Also, setInput() must be called for all inputs in the model.
     */
    GraphStatus run();

    /*
     * Set the input tensors for the model.
     * The input_name much match the input tensor name in the model.
     * NOTE: setInput() must be called for each input tensor before run() is called.
     */
    template <typename T>
    int setInput(std::string input_name, std::vector<T>& vals);

    /*
     * Retrieve the output tensors from the graph after running.
     * The output_name much match the output tensor name in the model.
     * NOTE: run() must be called before outputs are retrieved.
     */
    template <typename T>
    std::vector<T> getOutput(std::string output_name);

private:
    std::unique_ptr<ModelRunnerImpl> model_runner_impl;
};

};    // namespace TosaReference

#endif

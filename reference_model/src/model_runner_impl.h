
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

#ifndef MODEL_RUNNER_IMPL_H_
#define MODEL_RUNNER_IMPL_H_

#include "model_runner.h"
#include "graph_status.h"
#include "version.h"

#include "array_proxy.h"
#include "ops/op_factory.h"
#include "subgraph_traverser.h"
#include "tosa_serialization_handler.h"

namespace TosaReference
{

/*
 * This is a private implementation of the IModelRunner class.
 * See documented IModelRunner for usage.
 */
class ModelRunnerImpl
{
public:
    ModelRunnerImpl();
    ModelRunnerImpl(const func_config_t& func_config, const func_debug_t& func_debug);

    ~ModelRunnerImpl();

    void setFuncConfig(func_config_t& func_config);
    void setFuncDebug(func_debug_t& func_debug);

    GraphStatus initialize(TosaSerializationBasicBlock& bb);
    GraphStatus initialize(TosaSerializationHandler& serialization_handler);
    GraphStatus run();

    template <typename T>
    int setInput(std::string input_name, ArrayProxy<T> vals);
    int setInput(std::string input_name, uint8_t* raw_ptr, size_t size);

    template <typename T>
    std::vector<T> getOutput(std::string output_name);
    int getOutput(std::string output_name, uint8_t* ptr, size_t size);

private:
    SubgraphTraverser* _main_gt = nullptr;

    // Used to determine if all input tensors have been set correctly.
    uint32_t n_input_tensors = 0;

    GraphStatus initialize(TosaSerializationBasicBlock* bb, TosaSerializationHandler* serialization_handler);
    void validateTosaVersion(TosaSerializationHandler& serialization_handler);
    void checkGraphStatus(SubgraphTraverser& main_gt);
};

};    // namespace TosaReference

#endif

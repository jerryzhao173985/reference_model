
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

#include "model_runner_impl.h"

using namespace TosaReference;

// Global instantiation of configuration and debug objects
func_config_t g_func_config;
func_debug_t g_func_debug;

IModelRunner::IModelRunner() : model_runner_impl(new ModelRunnerImpl())
{}

IModelRunner::IModelRunner(const func_config_t& func_config,
                           const func_debug_t& func_debug)
    : model_runner_impl(new ModelRunnerImpl(func_config, func_debug))
{}

IModelRunner::~IModelRunner()
{}

void IModelRunner::setFuncConfig(func_config_t& func_config)
{
    model_runner_impl->setFuncConfig(func_config);
}

void IModelRunner::setFuncDebug(func_debug_t& func_debug)
{
    model_runner_impl->setFuncDebug(func_debug);
}

GraphStatus IModelRunner::initialize(tosa::TosaSerializationHandler& serialization_handler)
{
    return model_runner_impl->initialize(serialization_handler);
}

GraphStatus IModelRunner::run()
{
    return model_runner_impl->run();
}

template <typename T>
int IModelRunner::setInput(std::string input_name, std::vector<T>& vals)
{
    return model_runner_impl->setInput<T>(input_name, ArrayProxy(vals.size(), vals.data()));
}

int IModelRunner::setInput(std::string input_name, uint8_t* raw_ptr, size_t size)
{
    return model_runner_impl->setInput(input_name, raw_ptr, size);
}

template <typename T>
std::vector<T> IModelRunner::getOutput(std::string output_name)
{
    return model_runner_impl->getOutput<T>(output_name);
}

int IModelRunner::getOutput(std::string output_name, uint8_t* raw_ptr, size_t size)
{
    return model_runner_impl->getOutput(output_name, raw_ptr, size);
}

// Template explicit specialization
template int IModelRunner::setInput<float>(std::string input_name, std::vector<float>& vals);
template int IModelRunner::setInput<half_float::half>(std::string input_name, std::vector<half_float::half>& vals);
template int IModelRunner::setInput<int32_t>(std::string input_name, std::vector<int32_t>& vals);
template int IModelRunner::setInput<int64_t>(std::string input_name, std::vector<int64_t>& vals);
template int IModelRunner::setInput<unsigned char>(std::string input_name, std::vector<unsigned char>& vals);

template std::vector<float> IModelRunner::getOutput<float>(std::string output_name);
template std::vector<half_float::half> IModelRunner::getOutput<half_float::half>(std::string output_name);
template std::vector<int32_t> IModelRunner::getOutput<int32_t>(std::string output_name);
template std::vector<int64_t> IModelRunner::getOutput<int64_t>(std::string output_name);
template std::vector<unsigned char> IModelRunner::getOutput<unsigned char>(std::string output_name);
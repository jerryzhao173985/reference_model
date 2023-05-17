
// Copyright (c) 2022-2023, ARM Limited.
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

ModelRunnerImpl::ModelRunnerImpl()
{}

ModelRunnerImpl::ModelRunnerImpl(const func_config_t& func_config,
                                 const func_debug_t& func_debug)
{
    g_func_config = func_config;
    g_func_debug = func_debug;
}

ModelRunnerImpl::~ModelRunnerImpl()
{
    g_func_debug.fini_debug();
    delete _main_gt;
};

void ModelRunnerImpl::setFuncConfig(func_config_t& func_config)
{
    g_func_config = func_config;
}
void ModelRunnerImpl::setFuncDebug(func_debug_t& func_debug)
{
    g_func_debug = func_debug;
}

GraphStatus ModelRunnerImpl::initialize(TosaSerializationHandler& serialization_handler)
{
    validateTosaVersion(serialization_handler);
    return initialize(serialization_handler.GetMainRegion()->GetBlocks()[0], &serialization_handler);
}

GraphStatus ModelRunnerImpl::initialize(TosaSerializationBasicBlock& bb)
{
    return initialize(&bb, nullptr);
}

GraphStatus ModelRunnerImpl::run()
{
    if (_main_gt == nullptr)
    {
        FATAL_ERROR("ModelRunnerImpl hasn't been initialized, please invoke initialize() before run()");
    }

    if (g_func_config.validate_only)
    {
        goto done;
    }

    // Validate the number of inputs matches the
    if (static_cast<uint32_t>(_main_gt->getNumInputTensors()) != n_input_tensors)
    {
        FATAL_ERROR("The number of inputs (%d) does not equal the number of inputs in the model (%d). "
                    "setInput() must be called for each input.",
                    n_input_tensors, _main_gt->getNumInputTensors());
    }

    if (g_func_config.eval)
    {
        // evaluateAll() returns 1 if graph evaluation is forced to be terminated earlier.
        if (_main_gt->evaluateAll())
        {
            ASSERT_MSG(_main_gt->getGraphStatus() != GraphStatus::TOSA_VALID,
                       "Upon evaluateAll() returning 1, graph can not be VALID.");
        }
        else
        {
            ASSERT_MSG(_main_gt->getGraphStatus() == GraphStatus::TOSA_VALID ||
                           _main_gt->getGraphStatus() == GraphStatus::TOSA_UNPREDICTABLE,
                       "Upon evaluateAll() returning 0, graph can only be VALID/UNPREDICTABLE.");
        }

        // Only generate output tensor if graph is valid.
        if (_main_gt->getGraphStatus() == GraphStatus::TOSA_VALID)
        {
            // Make sure output tensor is evaluated and show its value
            int num_output_tensors = _main_gt->getNumOutputTensors();
            bool all_output_valid  = true;
            for (int i = 0; i < num_output_tensors; i++)
            {
                const Tensor* ct = _main_gt->getOutputTensor(i);
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
                _main_gt->dumpGraph(g_func_debug.func_debug_file);
                FATAL_ERROR(
                    "SubgraphTraverser \"main\" error: Output tensors are not all valid at the end of evaluation.");
            }
        }
    }

done:
    // Print status if not valid and do cleanup.
    checkGraphStatus(*_main_gt);
    g_func_debug.fini_debug();

    return _main_gt->getGraphStatus();
}

template <typename T>
int ModelRunnerImpl::setInput(std::string input_name, ArrayProxy<T> vals)
{
    if (_main_gt == nullptr)
    {
        FATAL_ERROR("ModelRunner hasn't been initialized, please invoke initialize() before setInput()");
    }

    Tensor* tensor;
    tensor = _main_gt->getInputTensorByName(input_name);

    if (!tensor)
    {
        WARNING("Unable to find input tensor %s", input_name.c_str());
        return 1;
    }

    if (!tensor->is_allocated())
    {
        WARNING("Tensor %s is not allocated before being initialized", tensor->getName().c_str());
        return 1;
    }

    if (tensor->readfromVector(vals))
    {
        WARNING("Unable to convert input tensor %s to Tensor", tensor->getName().c_str());
        return 1;
    }

    // Push ready consumers to the next node list
    for (auto gn : tensor->getConsumers())
    {
        if (gn->hasAllInputsReady() && !gn->getOnNextNodeList())
        {
            _main_gt->addToNextNodeList(gn);
        }
    }

    n_input_tensors++;
    return 0;
}

int ModelRunnerImpl::setInput(std::string input_name, uint8_t* raw_ptr, size_t size)
{
    if (_main_gt == nullptr)
    {
        FATAL_ERROR("ModelRunner hasn't been initialized, please invoke initialize() before setInput()");
    }

    Tensor* tensor;
    tensor = _main_gt->getInputTensorByName(input_name);

    if (!tensor)
    {
        WARNING("Unable to find input tensor %s", input_name.c_str());
        return 1;
    }

    int status = 0;
    switch (tensor->getDtype())
    {
        case TOSA_REF_TYPE_FP16: {
            auto typed_ptr     = reinterpret_cast<half_float::half*>(raw_ptr);
            const int elements = size / sizeof(half_float::half);
            status             = setInput(input_name, ArrayProxy(elements, typed_ptr));
            break;
        }
        case TOSA_REF_TYPE_FP32: {
            auto typed_ptr     = reinterpret_cast<float*>(raw_ptr);
            const int elements = size / sizeof(float);
            status             = setInput(input_name, ArrayProxy(elements, typed_ptr));
            break;
        }
        default:
            status = 1;
    }

    return status;
}

template <typename T>
std::vector<T> ModelRunnerImpl::getOutput(std::string output_name)
{
    if (_main_gt == nullptr)
    {
        FATAL_ERROR("ModelRunner hasn't been initialized, please invoke initialize() and run() before getOutput()");
    }

    Tensor* tensor;
    tensor = _main_gt->getOutputTensorByName(output_name);

    if (!tensor)
    {
        WARNING("Unable to find output tensor %s", output_name.c_str());
        return std::vector<T>();
    }

    std::vector<T> outputs(tensor->getElementCount());

    if (tensor->writeToVector(ArrayProxy<T>(outputs)))
    {
        WARNING("Unable to convert output tensor %s to vector", tensor->getName().c_str());
        return std::vector<T>();
    }

    return outputs;
}

int ModelRunnerImpl::getOutput(std::string output_name, uint8_t* raw_ptr, size_t size)
{
    if (_main_gt == nullptr)
    {
        FATAL_ERROR("ModelRunner hasn't been initialized, please invoke initialize() and run() before getOutput()");
    }

    Tensor* tensor;
    tensor = _main_gt->getOutputTensorByName(output_name);

    if (!tensor)
    {
        WARNING("Unable to find output tensor %s", output_name.c_str());
        return 1;
    }

    int status = 0;
    switch (tensor->getDtype())
    {
        case TOSA_REF_TYPE_FP16: {
            auto typed_ptr     = reinterpret_cast<half_float::half*>(raw_ptr);
            const int elements = size / sizeof(half_float::half);
            status             = tensor->writeToVector(ArrayProxy(elements, typed_ptr));
            break;
        }
        case TOSA_REF_TYPE_FP32: {
            auto typed_ptr     = reinterpret_cast<float*>(raw_ptr);
            const int elements = size / sizeof(float);
            status             = tensor->writeToVector(ArrayProxy(elements, typed_ptr));
            break;
        }
        case TOSA_REF_TYPE_BOOL: {
            auto typed_ptr     = reinterpret_cast<unsigned char*>(raw_ptr);
            const int elements = size / sizeof(unsigned char);
            status             = tensor->writeToVector(ArrayProxy(elements, typed_ptr));
            break;
        }
        default:
            status = 1;
    }
    if (status)
    {
        WARNING("Unable to convert output tensor %s to vector", tensor->getName().c_str());
        return 1;
    }

    return 0;
}

GraphStatus ModelRunnerImpl::initialize(TosaSerializationBasicBlock* bb,
                                        TosaSerializationHandler* serialization_handler)
{
    if (serialization_handler != nullptr)
        validateTosaVersion(*serialization_handler);

    // Make nullptr in case ModelRunnerImpl is being initialized again with a different graph.
    _main_gt = nullptr;
    _main_gt = new SubgraphTraverser(bb, serialization_handler, nullptr);

    if (_main_gt == nullptr)
    {
        WARNING("An error occurred when generating main graph traverser.");
        return GraphStatus::TOSA_ERROR;
    }

    if (_main_gt->initializeGraph())
    {
        WARNING("Unable to initialize main graph traverser.");
        return _main_gt->getGraphStatus();
    }

    if (_main_gt->linkTensorsAndNodes())
    {
        WARNING("Failed to link tensors and nodes");
        return _main_gt->getGraphStatus();
    }

    if (_main_gt->validateGraph())
    {
        WARNING("Failed to validate graph.");
        return _main_gt->getGraphStatus();
    }

    if (_main_gt->allocateTensor())
    {
        WARNING("Failed to allocate tensor.");
        return _main_gt->getGraphStatus();
    }

    return _main_gt->getGraphStatus();
}

void ModelRunnerImpl::validateTosaVersion(TosaSerializationHandler& serialization_handler)
{
    TosaVersion model_version(TOSA_REFERENCE_MODEL_VERSION_MAJOR,
                              TOSA_REFERENCE_MODEL_VERSION_MINOR,
                              TOSA_REFERENCE_MODEL_VERSION_PATCH,
                              TOSA_REFERENCE_MODEL_VERSION_DRAFT);

    TosaVersion::compat_t is_compat = model_version.is_compatible(serialization_handler.GetVersion());
    switch (is_compat)
    {
        case TosaVersion::compat_t::COMPLETELY_COMPATIBLE:
            break;
        case TosaVersion::compat_t::PARTIALLY_COMPATIBLE:
            WARNING("Reference model version %s is partially compatible with serializer version %s.",
                    model_version.to_string().c_str(), serialization_handler.GetVersion().to_string().c_str());
            break;
        case TosaVersion::compat_t::NOT_COMPATIBLE:
            FATAL_ERROR("Reference model version %s is not compatible with serializer version %s.",
                        model_version.to_string().c_str(), serialization_handler.GetVersion().to_string().c_str());
    }
}

void ModelRunnerImpl::checkGraphStatus(SubgraphTraverser& main_gt)
{
    switch (main_gt.getGraphStatus())
    {
        case GraphStatus::TOSA_VALID:
            // Result is valid.
            break;
        case GraphStatus::TOSA_UNPREDICTABLE:
            WARNING("Graph result: UNPREDICTABLE.");
            break;
        case GraphStatus::TOSA_ERROR:
            WARNING("Graph result: ERROR.");
            break;
        default:
            WARNING("Unknown graph status code=%d.", (int)main_gt.getGraphStatus());
    }
}

// Template explicit specialization
template int ModelRunnerImpl::setInput<float>(std::string input_name, ArrayProxy<float> vals);
template int ModelRunnerImpl::setInput<half_float::half>(std::string input_name, ArrayProxy<half_float::half> vals);
template int ModelRunnerImpl::setInput<int32_t>(std::string input_name, ArrayProxy<int32_t> vals);
template int ModelRunnerImpl::setInput<int64_t>(std::string input_name, ArrayProxy<int64_t> vals);
template int ModelRunnerImpl::setInput<unsigned char>(std::string input_name, ArrayProxy<unsigned char> vals);

template std::vector<float> ModelRunnerImpl::getOutput<float>(std::string output_name);
template std::vector<half_float::half> ModelRunnerImpl::getOutput<half_float::half>(std::string output_name);
template std::vector<int32_t> ModelRunnerImpl::getOutput<int32_t>(std::string output_name);
template std::vector<int64_t> ModelRunnerImpl::getOutput<int64_t>(std::string output_name);
template std::vector<unsigned char> ModelRunnerImpl::getOutput<unsigned char>(std::string output_name);
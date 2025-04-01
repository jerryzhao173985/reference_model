// Copyright (c) 2024-2025, ARM Limited.
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
#ifndef _TEST_UTILS_H
#define _TEST_UTILS_H

#include "cfloat.h"
#include "half.hpp"
#include "model_runner.h"
#include "tosa_generated.h"

// Remove conflicting REQUIRE definition between doctest and reference_model
#undef REQUIRE
#include "doctest/doctest.h"

using namespace ct;
using namespace tosa;
using namespace TosaReference;
using namespace half_float;

template <typename T>
constexpr DType NativeType2DType()
{
    if constexpr (std::is_same<T, bool>::value)
        return DType_BOOL;

    if constexpr (std::is_same<T, int8_t>::value)
        return DType_INT8;

    if constexpr (std::is_same<T, uint8_t>::value)
        return DType_INT8;

    if constexpr (std::is_same<T, int16_t>::value)
        return DType_INT16;

    if constexpr (std::is_same<T, uint16_t>::value)
        return DType_INT16;

    if constexpr (std::is_same<T, int32_t>::value)
        return DType_INT32;

    if constexpr (std::is_same<T, ct::binary16>::value)
        return DType_FP16;

    if constexpr (std::is_same<T, half_float::half>::value)
        return DType_FP16;

    if constexpr (std::is_same<T, float>::value)
        return DType_FP32;

    if constexpr (std::is_same<T, ct::binary32>::value)
        return DType_FP32;

    if constexpr (std::is_same<T, ct::bfloat16>::value)
        return DType_BF16;

    if constexpr (std::is_same<T, ct::fp8_e5m2>::value)
        return DType_FP8E5M2;

    if constexpr (std::is_same<T, ct::fp8_e4m3>::value)
        return DType_FP8E4M3;

    return DType_UNKNOWN;
}

/// \brief helper for building flatbuffers for unit testing purposes
class RefModelTestBuilder
{
private:
    TosaSerializationHandler _serHandler;
    TosaSerializationRegion* _mainRegion;
    TosaSerializationBasicBlock* _mainBlock;
    IModelRunner _modelRunner;
    // This will start at 0 and increase by 1 every time we use setInput.
    int _inputsAlreadySet;

public:
    RefModelTestBuilder()
        : RefModelTestBuilder(func_config_t{}, func_debug_t{})
    {}
    RefModelTestBuilder(func_config_t funcConfig, func_debug_t funcDebug)
        : _modelRunner(funcConfig, funcDebug)
        , _inputsAlreadySet(0)
    {
        // Create the main region
        _serHandler.GetRegions().push_back(std::make_unique<TosaSerializationRegion>("region_main"));
        _mainRegion = _serHandler.GetMainRegion();

        // Create the main block
        _mainRegion->GetBlocks().push_back(std::make_unique<TosaSerializationBasicBlock>("block_main", "region_main"));
        _mainBlock = _mainRegion->GetBlockByName("block_main");
    }

    void addInput(std::vector<int32_t> shape, DType dtype)
    {
        int previousSize      = _mainBlock->GetInputs().size();
        std::string inputName = getInputName(previousSize);
        _mainBlock->GetInputs().push_back(inputName);

        std::vector<uint8_t> data;    // empty data: these are not constants
        _mainBlock->GetTensors().push_back(
            std::make_unique<TosaSerializationTensor>(inputName, shape, dtype, data, false, false));
    }

    void addOutput(std::vector<int32_t> shape, DType dtype)
    {
        int previousSize       = _mainBlock->GetOutputs().size();
        std::string outputName = getOutputName(previousSize);
        _mainBlock->GetOutputs().push_back(outputName);

        std::vector<uint8_t> data;    // empty data: these are not constants
        _mainBlock->GetTensors().push_back(
            std::make_unique<TosaSerializationTensor>(outputName, shape, dtype, data, false, false));
    }

    // addOp should be used once all inputs and outputs have already been added.
    void addOp(Op op, Attribute attr_type, TosaAttributeBase* attr)
    {
        _mainBlock->GetOperators().push_back(std::make_unique<TosaSerializationOperator>(
            op, attr_type, attr, _mainBlock->GetInputs(), _mainBlock->GetOutputs()));
    }

    void initializeRunner()
    {
        // Initializing with the block directly skips some extra checks like version matching,
        // which we don't need
        _modelRunner.initialize(*_mainBlock);
    }

    template <typename T>
    void setInput(std::vector<T>& vals)
    {
        REQUIRE_MESSAGE(_inputsAlreadySet < _mainBlock->GetInputs().size(),
                        "Error in test setup: setting more inputs than the number that were added");
        // The vector-based call to setInput doesn't work with cfloat.h types.
        // But the raw-ptr one does, so use that one.
        uint8_t* data = reinterpret_cast<uint8_t*>(vals.data());
        _modelRunner.setInput(_mainBlock->GetInputs()[_inputsAlreadySet], data, vals.size() * sizeof(T));
        _inputsAlreadySet++;
    }

    std::string getInputName(int num)
    {
        return "input" + std::to_string(num);
    }

    std::string getOutputName(int num)
    {
        return "output" + std::to_string(num);
    }

    GraphStatus run()
    {
        return _modelRunner.run();
    }

    template <typename T>
    std::vector<T> getOutput(int i, size_t size)
    {
        std::vector<T> output(size);
        _modelRunner.getOutput(getOutputName(i), reinterpret_cast<uint8_t*>(output.data()), size * sizeof(T));
        return output;
    }
};

template <typename T>
void compareOutput(std::vector<T>& tensor1, std::vector<T>& tensor2)
{
    REQUIRE_MESSAGE(tensor1.size() == tensor2.size(), "Sizes of tensor 1 and tensor2 do not match");
    for (size_t i = 0; i < tensor1.size(); ++i)
    {
        CHECK_MESSAGE(tensor1[i] == doctest::Approx(tensor2[i]), "Difference in index ", i);
    }
}

template <typename T>
void compareOutputSpecial(std::vector<T>& tensor1, std::vector<T>& tensor2)
{
    REQUIRE_MESSAGE(tensor1.size() <= tensor2.size(),
                    "Test construction error: tensor1 must not be larger than tensor2");
    for (size_t i = 0; i < tensor1.size(); ++i)
    {
        T t1_i            = tensor1[i];
        T t2_i            = tensor2[i];
        const double d1_i = static_cast<double>(t1_i);
        const double d2_i = static_cast<double>(t2_i);

        if (std::isfinite(d1_i) && std::isfinite(d2_i))
        {
            INFO("index ", i);
            // If both values are zero, check that their sign bits match.
            if (d1_i == 0.0 && d2_i == 0.0)
            {
                // Check sign for 0
                CHECK(std::signbit(d1_i) == std::signbit(d2_i));
            }
            else
            {
                CHECK(t1_i == t2_i);
            }
        }
        else
        {
            INFO("index", i);

            CHECK(std::isnan(d1_i) == std::isnan(d2_i));
            // Do not check sign for NaNs
            if (std::isnan(d1_i) && std::isnan(d2_i))
                continue;

            CHECK(std::isinf(d1_i) == std::isinf(d2_i));
            CHECK(std::signbit(d1_i) == std::signbit(d2_i));
        }
    }
}

#endif

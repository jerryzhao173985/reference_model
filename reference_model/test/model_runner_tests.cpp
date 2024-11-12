
// Copyright (c) 2022,2024 ARM Limited.
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

#ifndef DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#endif

#include "cfloat.h"
#include "general_utils.h"
#include "model_runner.h"
#include "test_utils.h"

#include <cmath>
#include <numeric>

// Remove conflicting REQUIRE definition between doctest and reference_model
#undef REQUIRE

#include "doctest.h"

using namespace TosaReference;
using namespace tosa;
using namespace ct;
using namespace half_float;

template <typename T>
void compareOutput(std::vector<T>& tensor1, std::vector<T>& tensor2)
{
    REQUIRE_MESSAGE(tensor1.size() == tensor2.size(), "Sizes of tensor 1 and tensor2 do not match");
    for (size_t i = 0; i < tensor1.size(); ++i)
    {
        CHECK_MESSAGE(tensor1[i] == doctest::Approx(tensor2[i]), "");
    }
}

template <typename T>
void compareOutputSpecial(std::vector<T>& tensor1, std::vector<T>& tensor2)
{
    REQUIRE_MESSAGE(tensor1.size() <= tensor2.size(),
                    "Test construction error: tensor1 must not be larger than tensor2");
    for (size_t i = 0; i < tensor1.size(); ++i)
    {
        T t1_i = tensor1[i];
        T t2_i = tensor2[i];
        if (std::isfinite(t1_i) && std::isfinite(t2_i))
        {
            CHECK_MESSAGE(t2_i == t2_i, "");
        }
        else
        {
            bool specialCharacteristicsMatch = std::isinf(t1_i) == std::isinf(t2_i) &&
                                               std::signbit(t1_i) == std::signbit(t2_i) &&
                                               std::isnan(t1_i) == std::isnan(t2_i);
            CHECK_MESSAGE(specialCharacteristicsMatch, "");
        }
    }
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
        _modelRunner.setInput(_mainBlock->GetInputs()[_inputsAlreadySet], vals);
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
    std::vector<T> getOutput(int i)
    {
        return _modelRunner.getOutput<T>(getOutputName(i));
    }
};

TEST_SUITE("model_runner")
{

    TEST_CASE("simple_add_f32_test")
    {
        std::string test_root(std::string(PROJECT_ROOT) + "../examples/test_add_1x4x4x4_f32/");
        std::string tosa_model_file(test_root + "flatbuffer-tflite/test_add_1x4x4x4_f32.tosa");
        std::string input0_file(test_root + "placeholder_0.npy");
        std::string input1_file(test_root + "placeholder_1.npy");
        std::string expected_output_file(test_root + "tflite_result.npy");

        std::vector<std::string> input_names = { "TosaInput_0", "TosaInput_1" };
        std::string output_name              = "TosaOutput_0";

        std::vector<int32_t> input0_shape = { 1, 4, 4, 1 };
        std::vector<int32_t> input1_shape = { 1, 4, 4, 4 };
        std::vector<int32_t> output_shape = { 1, 4, 4, 4 };

        std::vector<std::vector<float>> inputs(input_names.size());
        std::vector<float> actual_outputs   = {};
        std::vector<float> expected_outputs = {};

        // Read in inputs and expected outputs.
        inputs[0]        = readFromNpyFile<float>(input0_file.c_str(), input0_shape);
        inputs[1]        = readFromNpyFile<float>(input1_file.c_str(), input1_shape);
        expected_outputs = readFromNpyFile<float>(expected_output_file.c_str(), output_shape);

        TosaSerializationHandler handler;
        tosa_err_t error = handler.LoadFileTosaFlatbuffer(tosa_model_file.c_str());
        CHECK((error == tosa::TOSA_OK));

        GraphStatus status;

        // Initialize the ModelRunner with configurations.
        IModelRunner runner;
        status = runner.initialize(handler);
        CHECK((status == GraphStatus::TOSA_VALID));

        runner.setInput(input_names[0], inputs[0]);
        runner.setInput(input_names[1], inputs[1]);

        // Run the ModelRunner using test inputs.
        status = runner.run();
        CHECK((status == GraphStatus::TOSA_VALID));

        actual_outputs = runner.getOutput<float>(output_name);
        CHECK(!actual_outputs.empty());

        compareOutput(expected_outputs, actual_outputs);
    }

    TEST_CASE("conv2d_f32_test")
    {
        std::string test_root(std::string(PROJECT_ROOT) +
                              "../examples/test_conv2d_1x1_1x32x32x8_f32_st11_padSAME_dilat11/");
        std::string tosa_model_file(test_root +
                                    "flatbuffer-tflite/test_conv2d_1x1_1x32x32x8_f32_st11_padSAME_dilat11.tosa");
        std::string input_file(test_root + "placeholder_0.npy");
        std::string expected_output_file(test_root + "tflite_result.npy");

        std::string input_name  = "TosaInput_0";
        std::string output_name = "TosaOutput_0";

        std::vector<int32_t> input_shape  = { 1, 32, 32, 8 };
        std::vector<int32_t> output_shape = { 1, 32, 32, 16 };

        // Read in inputs and expected outputs.
        std::vector<float> inputs           = readFromNpyFile<float>(input_file.c_str(), input_shape);
        std::vector<float> expected_outputs = readFromNpyFile<float>(expected_output_file.c_str(), output_shape);

        TosaSerializationHandler handler;
        tosa_err_t error = handler.LoadFileTosaFlatbuffer(tosa_model_file.c_str());
        CHECK((error == tosa::TOSA_OK));

        GraphStatus status;

        // Initialize the ModelRunner with configurations.
        IModelRunner runner;
        status = runner.initialize(handler);
        CHECK((status == GraphStatus::TOSA_VALID));

        runner.setInput(input_name, inputs);

        // Run the ModelRunner using test inputs.
        status = runner.run();
        CHECK((status == GraphStatus::TOSA_VALID));

        std::vector<float> actual_outputs = runner.getOutput<float>(output_name);
        CHECK(!actual_outputs.empty());

        compareOutput(expected_outputs, actual_outputs);
    }

    TEST_CASE("conv2d_f32_validate_only_test")
    {
        std::string test_root(std::string(PROJECT_ROOT) +
                              "../examples/test_conv2d_1x1_1x32x32x8_f32_st11_padSAME_dilat11/");
        std::string tosa_model_file(test_root +
                                    "flatbuffer-tflite/test_conv2d_1x1_1x32x32x8_f32_st11_padSAME_dilat11.tosa");

        TosaSerializationHandler handler;
        tosa_err_t error = handler.LoadFileTosaFlatbuffer(tosa_model_file.c_str());
        CHECK((error == tosa::TOSA_OK));

        GraphStatus status;
        func_debug_t funcDebug;

        func_config_t funcConfig;
        funcConfig.validate_only = 1;

        // Initialize the ModelRunner with configurations.
        IModelRunner runner = IModelRunner(funcConfig, funcDebug);
        runner.setFuncConfig(funcConfig);
        status = runner.initialize(handler);
        CHECK((status == GraphStatus::TOSA_VALID));

        // Run the ModelRunner using no inputs, as validate_only is specified run() should still work.
        status = runner.run();
        CHECK((status == GraphStatus::TOSA_VALID));
    }

}    // TEST_SUITE(model_runner)

TEST_SUITE("reference model FP SPECIAL")
{
    // TODO(ITL): apparently our model runner doesn't support half or fp8.
    // In the meantime this "template" only handles float.
    TEST_CASE_TEMPLATE("CAST FP SPECIAL", IN_FP_TYPE, float)
    {
        // NOTE: this is a small example included for the purpose of trying out
        // the RefModelTestBuilder.
        RefModelTestBuilder tb{};

        // TODO(ITL): tests combinations of types more thoroughly
        DType inDtype  = NativeType2Dtype<IN_FP_TYPE>();
        DType outDtype = DType_FP16;

        tb.addInput({ 5 }, inDtype);
        tb.addOutput({ 5 }, outDtype);

        tb.addOp(Op_CAST, Attribute_NONE, nullptr);

        tb.initializeRunner();

        std::vector<IN_FP_TYPE> inVals = {
            std::numeric_limits<IN_FP_TYPE>::infinity(),
            std::numeric_limits<IN_FP_TYPE>::infinity(),
            std::numeric_limits<IN_FP_TYPE>::quiet_NaN(),
            static_cast<IN_FP_TYPE>(+0.0),
            static_cast<IN_FP_TYPE>(-0.0),
        };

        std::vector<half> expectedOut = { std::numeric_limits<half>::infinity(), std::numeric_limits<half>::infinity(),
                                          std::numeric_limits<half>::quiet_NaN(), static_cast<half>(+0.0f),
                                          static_cast<half>(-0.0f) };

        tb.setInput(inVals);

        REQUIRE(tb.run() == GraphStatus::TOSA_VALID);
        std::vector<half> actualOut = tb.getOutput<half>(0);

        compareOutputSpecial(expectedOut, actualOut);
    }
}    // TEST_SUITE("reference model FP SPECIAL")

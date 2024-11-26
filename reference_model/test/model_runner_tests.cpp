
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

#include <cmath>
#include <numeric>

// Include this last because it redefines REQUIRE
#include "test_utils.h"

using namespace TosaReference;
using namespace tosa;
using namespace ct;
using namespace half_float;

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

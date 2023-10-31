
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

#ifndef DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#endif

#include "general_utils.h"
#include "model_runner.h"
#include "operators.h"

#include <numeric>

// Remove conflicting REQUIRE definition between doctest and reference_model
#undef REQUIRE

#include "doctest.h"

using namespace TosaReference;
using namespace tosa;

template <typename T>
void compareOutput(std::vector<T>& tensor1, std::vector<T>& tensor2, size_t size)
{
    for (size_t i = 0; i < size; ++i)
    {
        CHECK_MESSAGE(tensor1[i] == doctest::Approx(tensor2[i]), "");
    }
}

TEST_SUITE("model_runner")
{

    TEST_CASE("op_entry_add")
    {
        // Inputs/Outputs
        tosa_datatype_t dt                = tosa_datatype_fp32_t;
        std::vector<int32_t> input_shape  = { 2, 4, 4, 1 };
        std::vector<int32_t> output_shape = { 2, 4, 4, 1 };
        std::vector<float> srcData1(32, 4.0f);
        std::vector<float> srcData2(32, 3.0f);
        std::vector<float> dstData(32, 0.0f);

        tosa_tensor_t input1;
        input1.shape     = input_shape.data();
        input1.num_dims  = input_shape.size();
        input1.data_type = dt;
        input1.data      = reinterpret_cast<uint8_t*>(srcData1.data());
        input1.size      = srcData1.size() * sizeof(float);

        tosa_tensor_t input2;
        input2.shape     = input_shape.data();
        input2.num_dims  = input_shape.size();
        input2.data_type = dt;
        input2.data      = reinterpret_cast<uint8_t*>(srcData2.data());
        input2.size      = srcData2.size() * sizeof(float);

        tosa_tensor_t output;
        output.shape     = output_shape.data();
        output.num_dims  = output_shape.size();
        output.data_type = dt;
        output.data      = reinterpret_cast<uint8_t*>(dstData.data());
        output.size      = dstData.size() * sizeof(float);

        // Execution
        auto status = tosa_run_add(input1, input2, output, {});
        CHECK((status == tosa_status_valid));

        // Compare results
        std::vector<float> expectedData(8, 7.0f);
        compareOutput(dstData, expectedData, expectedData.size());
    }

    TEST_CASE("op_entry_avg_pool2d")
    {
        // Pool parameters
        const int32_t kernel[2] = { 2, 2 };
        const int32_t stride[2] = { 2, 2 };
        const int32_t pad[4]    = { 0, 0, 0, 0 };

        // Inputs/Outputs
        tosa_datatype_t dt                = tosa_datatype_fp32_t;
        std::vector<int32_t> input_shape  = { 2, 4, 4, 1 };
        std::vector<int32_t> output_shape = { 2, 2, 2, 1 };
        std::vector<float> srcData(32, 7.0f);
        std::vector<float> dstData(8, 0.f);

        tosa_tensor_t input;
        input.shape     = input_shape.data();
        input.num_dims  = input_shape.size();
        input.data_type = dt;
        input.data      = reinterpret_cast<uint8_t*>(srcData.data());
        input.size      = srcData.size() * sizeof(float);

        tosa_tensor_t output;
        output.shape     = output_shape.data();
        output.num_dims  = output_shape.size();
        output.data_type = dt;
        output.data      = reinterpret_cast<uint8_t*>(dstData.data());
        output.size      = dstData.size() * sizeof(float);

        tosa_acc_size_t acc_size = tosa_acc_size_fp32_t;

        // Execution
        auto status = tosa_run_avg_pool2d(input, kernel, stride, pad, acc_size, 0, 0, output, {});
        CHECK((status == tosa_status_valid));

        // Compare results
        std::vector<float> expectedData(8, 7.0f);
        compareOutput(dstData, expectedData, expectedData.size());
    }

    TEST_CASE("op_entry_conv2d")
    {
        // Conv parameters
        const int32_t stride[2]   = { 1, 1 };
        const int32_t pad[4]      = { 0, 0, 0, 0 };
        const int32_t dilation[2] = { 1, 1 };

        // Inputs/Outputs
        tosa_datatype_t dt                = tosa_datatype_fp32_t;
        std::vector<int32_t> input_shape  = { 1, 32, 32, 8 };
        std::vector<int32_t> output_shape = { 1, 32, 32, 16 };
        std::vector<int32_t> weight_shape = { 16, 1, 1, 8 };
        std::vector<int32_t> bias_shape   = { 16 };
        std::vector<float> srcData(32 * 32 * 8, 1.0f);
        std::vector<float> dstData(32 * 32 * 16, 0.f);
        std::vector<float> biasData(16, 0.f);
        std::vector<float> weightData(16 * 8, 1.0f);

        tosa_tensor_t input;
        input.shape     = input_shape.data();
        input.num_dims  = input_shape.size();
        input.data_type = dt;
        input.data      = reinterpret_cast<uint8_t*>(srcData.data());
        input.size      = srcData.size() * sizeof(float);

        tosa_tensor_t weight;
        weight.shape     = weight_shape.data();
        weight.num_dims  = weight_shape.size();
        weight.data_type = dt;
        weight.data      = reinterpret_cast<uint8_t*>(weightData.data());
        weight.size      = weightData.size() * sizeof(float);

        tosa_tensor_t bias;
        bias.shape     = bias_shape.data();
        bias.num_dims  = bias_shape.size();
        bias.data_type = dt;
        bias.data      = reinterpret_cast<uint8_t*>(biasData.data());
        bias.size      = biasData.size() * sizeof(float);

        tosa_tensor_t output;
        output.shape     = output_shape.data();
        output.num_dims  = output_shape.size();
        output.data_type = dt;
        output.data      = reinterpret_cast<uint8_t*>(dstData.data());
        output.size      = dstData.size() * sizeof(float);

        const int32_t input_zp  = 0;
        const int32_t weight_zp = 0;

        // Execution
        auto status = tosa_run_conv2d(input, weight, bias, pad, stride, dilation, input_zp, weight_zp, output, {});
        CHECK((status == tosa_status_valid));

        // Compare results
        std::vector<float> expectedData(32 * 32 * 16, 8.0f);
        compareOutput(dstData, expectedData, expectedData.size());
    }

    TEST_CASE("op_entry_conv2d_abs_mode")
    {
        // Conv parameters
        const int32_t stride[2]   = { 1, 1 };
        const int32_t pad[4]      = { 0, 0, 0, 0 };
        const int32_t dilation[2] = { 1, 1 };

        // Inputs/Outputs
        tosa_datatype_t dt                = tosa_datatype_fp32_t;
        std::vector<int32_t> input_shape  = { 1, 32, 32, 8 };
        std::vector<int32_t> output_shape = { 1, 32, 32, 16 };
        std::vector<int32_t> weight_shape = { 16, 1, 1, 8 };
        std::vector<int32_t> bias_shape   = { 16 };
        std::vector<float> srcData(32 * 32 * 8, -1.0f);
        std::vector<float> dstData(32 * 32 * 16, 0.f);
        std::vector<float> biasData(16, 0.f);
        std::vector<float> weightData(16 * 8, 1.0f);

        tosa_tensor_t input;
        input.shape     = input_shape.data();
        input.num_dims  = input_shape.size();
        input.data_type = dt;
        input.data      = reinterpret_cast<uint8_t*>(srcData.data());
        input.size      = srcData.size() * sizeof(float);

        tosa_tensor_t weight;
        weight.shape     = weight_shape.data();
        weight.num_dims  = weight_shape.size();
        weight.data_type = dt;
        weight.data      = reinterpret_cast<uint8_t*>(weightData.data());
        weight.size      = weightData.size() * sizeof(float);

        tosa_tensor_t bias;
        bias.shape     = bias_shape.data();
        bias.num_dims  = bias_shape.size();
        bias.data_type = dt;
        bias.data      = reinterpret_cast<uint8_t*>(biasData.data());
        bias.size      = biasData.size() * sizeof(float);

        tosa_tensor_t output;
        output.shape     = output_shape.data();
        output.num_dims  = output_shape.size();
        output.data_type = dt;
        output.data      = reinterpret_cast<uint8_t*>(dstData.data());
        output.size      = dstData.size() * sizeof(float);

        const int32_t input_zp  = 0;
        const int32_t weight_zp = 0;

        // Execution
        func_ctx_t func_ctx;
        func_ctx.func_config.abs_mode = true;
        auto status =
            tosa_run_conv2d(input, weight, bias, pad, stride, dilation, input_zp, weight_zp, output, func_ctx);
        CHECK((status == tosa_status_valid));

        // Compare results
        std::vector<float> expectedData(32 * 32 * 16, 8.0f);
        compareOutput(dstData, expectedData, expectedData.size());
    }

    TEST_CASE("op_entry_max_pool2d")
    {
        // Pool parameters
        const int32_t kernel[2] = { 2, 2 };
        const int32_t stride[2] = { 2, 2 };
        const int32_t pad[4]    = { 0, 0, 0, 0 };

        // Inputs/Outputs
        tosa_datatype_t dt                = tosa_datatype_fp32_t;
        std::vector<int32_t> input_shape  = { 2, 4, 4, 1 };
        std::vector<int32_t> output_shape = { 2, 2, 2, 1 };
        std::vector<float> srcData(32);
        std::vector<float> dstData(8, 0.f);
        std::iota(std::begin(srcData), std::end(srcData), 1);

        tosa_tensor_t input;
        input.shape     = input_shape.data();
        input.num_dims  = input_shape.size();
        input.data_type = dt;
        input.data      = reinterpret_cast<uint8_t*>(srcData.data());
        input.size      = srcData.size() * sizeof(float);

        tosa_tensor_t output;
        output.shape     = output_shape.data();
        output.num_dims  = output_shape.size();
        output.data_type = dt;
        output.data      = reinterpret_cast<uint8_t*>(dstData.data());
        output.size      = dstData.size() * sizeof(float);

        // Execution
        auto status = tosa_run_max_pool2d(input, kernel, stride, pad, 0, 0, output, {});
        CHECK((status == tosa_status_valid));

        // Compare results
        std::vector<float> expectedData = { 6, 8, 14, 16, 22, 24, 30, 32 };
        compareOutput(dstData, expectedData, expectedData.size());
    }

    TEST_CASE("op_entry_pad")
    {
        // Inputs/Outputs
        tosa_datatype_t dt                 = tosa_datatype_fp32_t;
        std::vector<int32_t> input_shape   = { 2, 2 };
        std::vector<int32_t> padding_shape = { 1, 4 };
        std::vector<int32_t> output_shape  = { 4, 4 };
        std::vector<float> srcData1(4, 4.0f);
        std::vector<int32_t> padData(4, 1);
        std::vector<float> dstData(16, 0.0f);

        tosa_tensor_t input1;
        input1.shape     = input_shape.data();
        input1.num_dims  = input_shape.size();
        input1.data_type = dt;
        input1.data      = reinterpret_cast<uint8_t*>(srcData1.data());
        input1.size      = srcData1.size() * sizeof(float);

        tosa_tensor_t padding;
        padding.shape     = padding_shape.data();
        padding.num_dims  = padding_shape.size();
        padding.data_type = tosa_datatype_int32_t;
        padding.data      = reinterpret_cast<uint8_t*>(padData.data());
        padding.size      = padData.size() * sizeof(int32_t);

        tosa_tensor_t output;
        output.shape     = output_shape.data();
        output.num_dims  = output_shape.size();
        output.data_type = dt;
        output.data      = reinterpret_cast<uint8_t*>(dstData.data());
        output.size      = dstData.size() * sizeof(float);

        // Execution
        int32_t pad_const_int = 0;
        float pad_const_fp    = 5.0f;
        auto status           = tosa_run_pad(input1, padding, pad_const_int, pad_const_fp, output, func_ctx_t{});
        CHECK((status == tosa_status_valid));

        // Compare results
        // Expect a 4x4 array with a border of 5's and inner 2x2 of 4's
        std::vector<float> expectedData(16, 5.0f);
        expectedData[5]  = 4.0f;
        expectedData[6]  = 4.0f;
        expectedData[9]  = 4.0f;
        expectedData[10] = 4.0f;
        compareOutput(dstData, expectedData, expectedData.size());
    }

    TEST_CASE("op_entry_reshape")
    {
        // Inputs/Outputs
        tosa_datatype_t dt                = tosa_datatype_fp32_t;
        std::vector<int32_t> input_shape  = { 2, 2 };
        std::vector<int32_t> new_shape    = { 1, 2 };
        std::vector<int32_t> output_shape = { 4, 1 };
        std::vector<float> srcData1(4, 4.0f);
        std::vector<int32_t> shapeData = { 4, 1 };
        std::vector<float> dstData(4, 0.0f);

        tosa_tensor_t input1;
        input1.shape     = input_shape.data();
        input1.num_dims  = input_shape.size();
        input1.data_type = dt;
        input1.data      = reinterpret_cast<uint8_t*>(srcData1.data());
        input1.size      = srcData1.size() * sizeof(float);

        tosa_tensor_t shape;
        shape.shape     = new_shape.data();
        shape.num_dims  = new_shape.size();
        shape.data_type = tosa_datatype_int32_t;
        shape.data      = reinterpret_cast<uint8_t*>(shapeData.data());
        shape.size      = shapeData.size() * sizeof(int32_t);

        tosa_tensor_t output;
        output.shape     = output_shape.data();
        output.num_dims  = output_shape.size();
        output.data_type = dt;
        output.data      = reinterpret_cast<uint8_t*>(dstData.data());
        output.size      = dstData.size() * sizeof(float);

        // Execution
        auto status = tosa_run_reshape(input1, shape, output, func_ctx_t{});
        CHECK((status == tosa_status_valid));

        // Compare results
        std::vector<float> expectedData(4, 4.0f);
        compareOutput(dstData, expectedData, expectedData.size());
    }

    TEST_CASE("op_entry_tile")
    {
        // Inputs/Outputs
        tosa_datatype_t dt                   = tosa_datatype_fp32_t;
        std::vector<int32_t> input_shape     = { 2, 3 };
        std::vector<int32_t> multiples_shape = { 1, 2 };
        std::vector<int32_t> output_shape    = { 2, 6 };
        std::vector<float> srcData1          = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
        std::vector<int32_t> multiples_data  = { 1, 2 };
        std::vector<float> dstData(12, 0.0f);

        tosa_tensor_t input1;
        input1.shape     = input_shape.data();
        input1.num_dims  = input_shape.size();
        input1.data_type = dt;
        input1.data      = reinterpret_cast<uint8_t*>(srcData1.data());
        input1.size      = srcData1.size() * sizeof(float);

        tosa_tensor_t multiples;
        multiples.shape     = multiples_shape.data();
        multiples.num_dims  = multiples_shape.size();
        multiples.data_type = tosa_datatype_int32_t;
        multiples.data      = reinterpret_cast<uint8_t*>(multiples_data.data());
        multiples.size      = multiples_data.size() * sizeof(int32_t);

        tosa_tensor_t output;
        output.shape     = output_shape.data();
        output.num_dims  = output_shape.size();
        output.data_type = dt;
        output.data      = reinterpret_cast<uint8_t*>(dstData.data());
        output.size      = dstData.size() * sizeof(float);

        // Execution
        auto status = tosa_run_tile(input1, multiples, output, {});
        CHECK((status == tosa_status_valid));

        // Compare results
        std::vector<float> expectedData = { 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 4.0, 5.0, 6.0 };
        compareOutput(dstData, expectedData, expectedData.size());
    }

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

        compareOutput(expected_outputs, actual_outputs, expected_outputs.size());
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

        compareOutput(expected_outputs, actual_outputs, expected_outputs.size());
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

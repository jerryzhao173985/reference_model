
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

#include "general_utils.h"
#include "model_runner.h"

int main()
{
    using namespace TosaReference;

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
    std::vector<float> expected_outputs = {};
    std::vector<float> actual_outputs   = {};

    // Read in inputs and expected outputs for sample purposes.
    inputs[0]        = readFromNpyFile<float>(input0_file.c_str(), input0_shape);
    inputs[1]        = readFromNpyFile<float>(input1_file.c_str(), input1_shape);
    expected_outputs = readFromNpyFile<float>(expected_output_file.c_str(), output_shape);

    tosa::TosaSerializationHandler handler;
    tosa::tosa_err_t error = handler.LoadFileTosaFlatbuffer(tosa_model_file.c_str());
    if (error != tosa::TOSA_OK)
    {
        WARNING("An error occurred while loading the model from file.");
        return 1;
    }
    GraphStatus status;

    // Initialize the ModelRunner with configurations.
    IModelRunner runner;
    status = runner.initialize(handler);
    if (status != GraphStatus::TOSA_VALID)
    {
        WARNING("An error occurred while initializing.");
        return 1;
    }

    // Set the model inputs using the input names and input data.
    runner.setInput(input_names[0], inputs[0]);
    runner.setInput(input_names[1], inputs[1]);

    // Run the ModelRunner using test inputs.
    status = runner.run();
    if (status != GraphStatus::TOSA_VALID)
    {
        WARNING("An error occurred when running the model.");
        return 1;
    }

    // Get the outputs from the model.
    actual_outputs = runner.getOutput<float>(output_name);

    // Compare the actual output to the expected output.
    bool if_accurate = true;
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        if (actual_outputs[i] != expected_outputs[i])
        {
            WARNING("Actual output (%f) doesn't match expected output (%f).");
            if_accurate = false;
        }
    }

    if (!if_accurate)
    {
        WARNING("There were mismatches in actual vs expected output, see above output for more details.");
        return 1;
    }

    printf("The model ran successfully without errors and matched the expected output.\n");
    return 0;
}
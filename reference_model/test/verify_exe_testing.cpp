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

#include "doctest.h"
#include "tensor.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#ifdef __unix__
#include <sys/wait.h>
#endif

#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

//Test ToDo List
// - test for different data types
// - test with bnd result file

#ifdef __unix__    // Code will only be compiled on Unix-based systems
std::string getVerifyFileLocation()
{
    char res[PATH_MAX];
#if defined(__linux__)
    // Linux specific function to get exe path to tosa_verify
    ssize_t count = readlink("/proc/self/exe", res, PATH_MAX);
#elif defined(__APPLE__)
    // macOS equivalent to retrieve the executable path
    uint32_t count = PATH_MAX;
    if (_NSGetExecutablePath(res, &count) < 0)
    {
        throw std::runtime_error("Could not retrieve executable path");
    }
#else
#error Unknown OS
#endif
    std::string exe_path = std::string(res, (count > 0) ? count : 0);

    int adjust = std::string("/unit_tests").length();
    return exe_path.substr(0, exe_path.length() - adjust) + "/verify/tosa_verify";
}
#endif

// Helper function to create a mock file with specific content
void createMockFile(const std::string& filename, const std::string& content)
{
    std::ofstream file(filename);
    file << content;
    file.close();
}

// Helper function to remove the mock file after testing
void removeMockFile(const std::string& filename)
{
    std::remove(filename.c_str());
}

TEST_SUITE_BEGIN("verify_exe_tests");

TEST_CASE("missing and invalid")
{
#ifdef __unix__    // Code will only be compiled on Unix-based systems
    std::string verif_exe_path = getVerifyFileLocation();

    SUBCASE("valid exe")
    {
        //checks if the path to the tosa_verify is valid

        bool check = std::filesystem::exists(verif_exe_path) && std::filesystem::is_regular_file(verif_exe_path);

        REQUIRE(check);
    }

    SUBCASE("missing config file")
    {

        std::string command = verif_exe_path + " --test_desc";

        int result   = std::system(command.c_str());
        int exitCode = WEXITSTATUS(result);
        REQUIRE(exitCode == 1);
    }

    SUBCASE("invalid json in config file")
    {

        // Create a mock invalid JSON config file
        std::string invalidConfigFile = "invalid_config.json";
        createMockFile(invalidConfigFile, "{invalid json}");

        std::string refResultFile = "ref.npy";
        std::string impResultFile = "imp.npy";

        std::vector<int32_t> shape = { 2, 12, 10, 7 };

        const auto elements = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<int64_t>());
        std::vector<double> refData = { 5.1, 10.1, 3.0 };
        refData.resize(elements);
        std::vector<float> impData(refData.begin(), refData.end());

        NumpyUtilities::writeToNpyFile(refResultFile.c_str(), shape, refData.data());
        NumpyUtilities::writeToNpyFile(impResultFile.c_str(), shape, impData.data());

        std::string command = verif_exe_path + " --test_desc " + invalidConfigFile + " --imp_result_file " +
                              impResultFile + " --ref_result_file " + refResultFile;

        int result = std::system(command.c_str());

        int exitCode = WEXITSTATUS(result);
        REQUIRE(exitCode == 1);

        // Clean up
        removeMockFile(invalidConfigFile);
        removeMockFile(refResultFile);
        removeMockFile(impResultFile);
    }

    SUBCASE("unsupported data type")
    {
        // Create a mock config file with unsupported data type
        std::string unsupportedConfigFile = "unsupported_data_type.json";
        createMockFile(unsupportedConfigFile, R"({
            "ofm_name": [
                "output"
            ],
            "meta": {
                "compliance": {
                    "tensors": {
                        "output": {
                            "shape": [1, 2],
                            "data_type": "unsupported_type"
                        }
                    }
                }
            }
        })");

        std::string refResultFile = "ref.npy";
        std::string impResultFile = "imp.npy";

        std::vector<int32_t> shape = { 2, 12, 10, 7 };

        const auto elements = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<int64_t>());
        std::vector<double> refData = { 5.1, 10.1, 3.0 };
        refData.resize(elements);
        std::vector<float> impData(refData.begin(), refData.end());

        NumpyUtilities::writeToNpyFile(refResultFile.c_str(), shape, refData.data());
        NumpyUtilities::writeToNpyFile(impResultFile.c_str(), shape, impData.data());

        std::string command = verif_exe_path + " --test_desc " + unsupportedConfigFile + " --imp_result_file " +
                              impResultFile + " --ref_result_file " + refResultFile;

        int result = std::system(command.c_str());

        int exitCode = WEXITSTATUS(result);
        REQUIRE(exitCode == 1);

        // Clean up
        removeMockFile(unsupportedConfigFile);
        removeMockFile(refResultFile);
        removeMockFile(impResultFile);
    }

    SUBCASE("ofmName mismatch")
    {
        // Create a mock valid config file and result files
        std::string validConfigFile = "valid_config.json";

        std::string refResultFile = "ref.npy";
        std::string impResultFile = "imp.npy";

        std::vector<int32_t> shape = { 35, 32 };
        const auto elements        = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<int64_t>());
        std::vector<double> refData = { 5.1, 10.1, 3.0 };
        refData.resize(elements);
        std::vector<float> impData(refData.begin(), refData.end());

        NumpyUtilities::writeToNpyFile(refResultFile.c_str(), shape, refData.data());
        NumpyUtilities::writeToNpyFile(impResultFile.c_str(), shape, impData.data());

        createMockFile(validConfigFile, R"({
            "ofm_name": [
                "result-0"
            ],
            "meta": {
                "compliance": {
                    "version": "0.1",
                    "tensors": {
                        "result-0": {
                            "mode": "ULP",
                            "data_type": "FP32",
                            "shape": [35,32],
                            "ulp_info": {
                                "ulp": 0.5
                            }
                        }
                    }
                }
            }
        })");

        std::string command = verif_exe_path + " --test_desc " + validConfigFile + " --imp_result_file " +
                              impResultFile + " --ref_result_file " + refResultFile + " --ofm_name output";

        int result = std::system(command.c_str());

        int exitCode = WEXITSTATUS(result);
        REQUIRE(exitCode == 1);

        // Clean up
        removeMockFile(validConfigFile);
        removeMockFile(refResultFile);
        removeMockFile(impResultFile);
    }

    SUBCASE("multiple ofmName in config")
    {
        // Create a mock valid config file and result files
        std::string validConfigFile = "valid_config.json";

        std::string refResultFile = "ref.npy";
        std::string impResultFile = "imp.npy";

        std::vector<int32_t> shape = { 35, 32 };
        const auto elements        = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<int64_t>());
        std::vector<double> refData = { 5.1, 10.1, 3.0 };
        refData.resize(elements);
        std::vector<float> impData(refData.begin(), refData.end());

        NumpyUtilities::writeToNpyFile(refResultFile.c_str(), shape, refData.data());
        NumpyUtilities::writeToNpyFile(impResultFile.c_str(), shape, impData.data());

        createMockFile(validConfigFile, R"({
            "ofm_name": [
                "result-0",
                "result"
            ],
            "meta": {
                "compliance": {
                    "version": "0.1",
                    "tensors": {
                        "result-0": {
                            "mode": "ULP",
                            "data_type": "FP32",
                            "shape": [35,32],
                            "ulp_info": {
                                "ulp": 0.5
                            }
                        }
                    }
                }
            }
        })");

        std::string command = verif_exe_path + " --test_desc " + validConfigFile + " --imp_result_file " +
                              impResultFile + " --ref_result_file " + refResultFile;

        int result = std::system(command.c_str());

        int exitCode = WEXITSTATUS(result);
        REQUIRE(exitCode == 1);

        // Clean up
        removeMockFile(validConfigFile);
        removeMockFile(refResultFile);
        removeMockFile(impResultFile);
    }

    SUBCASE("correct usage of ofmName")
    {
        // Create a mock valid config file and result files
        std::string validConfigFile = "valid_config.json";
        std::string refResultFile   = "ref.npy";
        std::string impResultFile   = "imp.npy";

        std::vector<int32_t> shape = { 35, 32 };
        const auto elements        = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<int64_t>());
        std::vector<double> refData = { 5.1, 10.1, 3.0 };
        refData.resize(elements);
        std::vector<float> impData(refData.begin(), refData.end());

        NumpyUtilities::writeToNpyFile(refResultFile.c_str(), shape, refData.data());
        NumpyUtilities::writeToNpyFile(impResultFile.c_str(), shape, impData.data());

        createMockFile(validConfigFile, R"({
            "ofm_name": [
                "_"
            ],
            "meta": {
                "compliance": {
                    "version": "0.1",
                    "tensors": {
                        "result-0": {
                            "mode": "ULP",
                            "data_type": "FP32",
                            "shape": [35,32],
                            "ulp_info": {
                                "ulp": 0.5
                            }
                        }
                    }
                }
            }
        })");

        std::string command = verif_exe_path + " --test_desc " + validConfigFile + " --imp_result_file " +
                              impResultFile + " --ref_result_file " + refResultFile + " --ofm_name result-0";

        int result = std::system(command.c_str());

        int exitCode = WEXITSTATUS(result);
        REQUIRE(exitCode == 0);

        // Clean up
        removeMockFile(validConfigFile);
        removeMockFile(refResultFile);
        removeMockFile(impResultFile);
    }
    SUBCASE("mismatching shapes")
    {
        // Create a mock config file with mismatching shapes
        std::string mismatchingShapesConfigFile = "mismatching_shapes.json";
        createMockFile(mismatchingShapesConfigFile, R"({
            "ofm_name": [
                "output"
            ],
            "meta": {
                "compliance": {
                    "tensors": {
                        "output": {
                            "shape": [1, 2],
                            "data_type": "int32"
                        }
                    }
                }
            }
        })");

        std::string refResultFile = "ref.npy";
        std::string impResultFile = "imp.npy";

        std::vector<int32_t> shape = { 35, 32 };
        const auto elements        = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<int64_t>());
        std::vector<double> refData = { 5.1, 10.1, 3.0 };
        refData.resize(elements);
        std::vector<float> impData(refData.begin(), refData.end());

        NumpyUtilities::writeToNpyFile(refResultFile.c_str(), shape, refData.data());
        NumpyUtilities::writeToNpyFile(impResultFile.c_str(), shape, impData.data());

        std::string command = verif_exe_path + " --test_desc " + mismatchingShapesConfigFile + " --imp_result_file " +
                              impResultFile + " --ref_result_file " + refResultFile;

        int result = std::system(command.c_str());

        int exitCode = WEXITSTATUS(result);
        REQUIRE(exitCode == 1);

        // Clean up
        removeMockFile(mismatchingShapesConfigFile);
        removeMockFile(refResultFile);
        removeMockFile(impResultFile);
    }

    SUBCASE("missing tensor data in implementation")
    {
        std::string missingTensorDataConfigFile = "missing_tensor_data.json";
        createMockFile(missingTensorDataConfigFile, R"({
            "ofm_name": [
                "result-0"
            ],
            "meta": {
                "compliance": {
                    "version": "0.1",
                    "tensors": {
                        "result-0": {
                            "mode": "ULP",
                            "data_type": "FP32",
                            "shape": [35,32],
                            "ulp_info": {
                                "ulp": 0.5
                            }
                        }
                    }
                }
            }
        })");

        std::string refResultFile = "ref.npy";

        std::vector<int32_t> shape = { 35, 32 };

        const auto elements = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<int64_t>());
        std::vector<double> refData = { 5.1, 10.1, 3.0 };
        refData.resize(elements);

        NumpyUtilities::writeToNpyFile(refResultFile.c_str(), shape, refData.data());

        std::string command = verif_exe_path + " --test_desc " + missingTensorDataConfigFile +
                              " --imp_result_file "
                              "imp.npy" +
                              " --ref_result_file " + refResultFile;

        int result = std::system(command.c_str());

        int exitCode = WEXITSTATUS(result);
        REQUIRE(exitCode == 1);
        // Clean up
        removeMockFile(missingTensorDataConfigFile);
        removeMockFile(refResultFile);
    }

    SUBCASE("missing tensor data in reference")
    {
        std::string missingTensorDataConfigFile = "missing_tensor_data.json";
        createMockFile(missingTensorDataConfigFile, R"({
            "ofm_name": [
                "result-0"
            ],
            "meta": {
                "compliance": {
                    "version": "0.1",
                    "tensors": {
                        "result-0": {
                            "mode": "ULP",
                            "data_type": "FP32",
                            "shape": [35,32],
                            "ulp_info": {
                                "ulp": 0.5
                            }
                        }
                    }
                }
            }
        })");

        std::string impResultFile = "imp.npy";

        std::vector<int32_t> shape = { 35, 32 };

        const auto elements        = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<int64_t>());
        std::vector<float> impData = { 5.1, 10.1, 3.0 };
        impData.resize(elements);

        NumpyUtilities::writeToNpyFile(impResultFile.c_str(), shape, impData.data());

        std::string command = verif_exe_path + " --test_desc " + missingTensorDataConfigFile + " --imp_result_file " +
                              impResultFile +
                              " --ref_result_file "
                              "ref.npy";

        int result = std::system(command.c_str());

        int exitCode = WEXITSTATUS(result);
        REQUIRE(exitCode == 1);
        // Clean up
        removeMockFile(missingTensorDataConfigFile);
        removeMockFile(impResultFile);
    }
#endif
}

#ifdef __unix__    // Code will only be compiled on Unix-based systems
TEST_CASE("different types")
{
    std::string verif_exe_path = getVerifyFileLocation();

    SUBCASE("fp32")
    {
        // Create a mock valid config file and result files
        std::string validConfigFile = "valid_config.json";
        std::string refResultFile   = "ref.npy";
        std::string impResultFile   = "imp.npy";

        std::vector<int32_t> shape = { 35, 32 };
        const auto elements        = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<int64_t>());
        std::vector<double> refData = { 5.1, 10.1, 3.0 };
        refData.resize(elements);
        std::vector<float> impData(refData.begin(), refData.end());

        NumpyUtilities::writeToNpyFile(refResultFile.c_str(), shape, refData.data());
        NumpyUtilities::writeToNpyFile(impResultFile.c_str(), shape, impData.data());

        createMockFile(validConfigFile, R"({
            "ofm_name": [
                "result-0"
            ],
            "meta": {
                "compliance": {
                    "version": "0.1",
                    "tensors": {
                        "result-0": {
                            "mode": "ULP",
                            "data_type": "FP32",
                            "shape": [35,32],
                            "ulp_info": {
                                "ulp": 0.5
                            }
                        }
                    }
                }
            }
        })");

        std::string command = verif_exe_path + " --test_desc " + validConfigFile + " --imp_result_file " +
                              impResultFile + " --ref_result_file " + refResultFile;

        int result = std::system(command.c_str());

        int exitCode = WEXITSTATUS(result);
        REQUIRE(exitCode == 0);

        // Clean up
        removeMockFile(validConfigFile);
        removeMockFile(refResultFile);
        removeMockFile(impResultFile);
    }

    SUBCASE("int32")
    {
        // Create a mock valid config file and result files dynamically
        std::string validConfigFile = "valid_config_int32.json";
        std::string refResultFile   = "ref_int32.npy";
        std::string impResultFile   = "imp_int32.npy";

        std::vector<int32_t> shape = { 2, 12, 10, 7 };
        const auto elements        = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<int64_t>());
        std::vector<int32_t> tensorData = { 5, 10, 3 };
        tensorData.resize(elements);

        NumpyUtilities::writeToNpyFile(refResultFile.c_str(), shape, tensorData.data());
        NumpyUtilities::writeToNpyFile(impResultFile.c_str(), shape, tensorData.data());

        // Create the config file dynamically
        createMockFile(validConfigFile, R"({
            "ofm_name": [
                "result-0"
            ],
            "meta": {
                "compliance": {
                "version": "0.1",
                "tensors": {
                    "result-0": {
                    "mode": "EXACT",
                    "data_type": "INT32",
                    "shape": [
                        2,
                        12,
                        10,
                        7
                    ]
                    }
                }
                }
            }
        })");

        // Run the verify command with dynamically generated files
        std::string command = verif_exe_path + " --test_desc " + validConfigFile + " --imp_result_file " +
                              impResultFile + " --ref_result_file " + refResultFile;

        int result = std::system(command.c_str());

        int exitCode = WEXITSTATUS(result);
        REQUIRE(exitCode == 0);

        // Clean up dynamically generated files
        removeMockFile(validConfigFile);
        removeMockFile(refResultFile);
        removeMockFile(impResultFile);
    }
}
#endif
TEST_SUITE_END();

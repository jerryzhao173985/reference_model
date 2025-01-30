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
#include "verify.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#ifdef __unix__
#include <sys/wait.h>
#endif

#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

// TODO:
// - test for more data types
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
#endif    // __linux__
    std::string exe_path = std::string(res, (count > 0) ? count : 0);

    int adjust = std::string("/unit_tests").length();
    return exe_path.substr(0, exe_path.length() - adjust) + "/verify/tosa_verify";
}
#endif    // __unix__

TEST_SUITE_BEGIN("verify_exe_tests");

// Code will only be compiled on Unix-based systems, currently no testing
// on other platforms
#ifdef __unix__
TEST_CASE("Executable exists")
{
    std::string verif_exe_path = getVerifyFileLocation();

    SUBCASE("Test valid exe")
    {
        //checks if the path to the tosa_verify is valid
        bool check = std::filesystem::exists(verif_exe_path) && std::filesystem::is_regular_file(verif_exe_path);

        REQUIRE(check);
    }
}

// Helper function to create a mock Json file with specific content
void createMockJsonFile(const std::string& filename, const std::string& content)
{
    std::ofstream file(filename);
    file << content;
    file.close();
}

// Helper function to create mock Numpy result files, it will resize the
// incoming refData to match the shape size
template <typename refT, typename impT>
void createMockNumpyFiles(const std::string& refName,
                          const std::string& impName,
                          const std::vector<int32_t>& shape,
                          std::vector<refT>& refData)
{
    const auto elements = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<int64_t>());
    // Make sure we have the right amount of data
    refData.resize(elements);
    // Create a copy of the data in the implementation data type
    std::vector<impT> impData(refData.begin(), refData.end());

    NumpyUtilities::writeToNpyFile(refName.c_str(), shape, refData.data());
    NumpyUtilities::writeToNpyFile(impName.c_str(), shape, impData.data());
}

// Helper function to remove the mock file after testing
void removeMockFile(const std::string& filename)
{
    std::remove(filename.c_str());
}

// Helper function to run the verify exe with optional ofm_name option
int runVerifyCommand(const std::string& exePath,
                     const std::string& testDesc,
                     const std::string& refName,
                     const std::string& impName,
                     const std::optional<std::string>& ofmName = std::nullopt)
{
    std::string command =
        exePath + " --test_desc " + testDesc + " --imp_result_file " + impName + " --ref_result_file " + refName;
    if (ofmName)
    {
        command += " --ofm_name " + ofmName.value();
    }

    std::cout << "Command: " << command << std::endl;
    int result = std::system(command.c_str());
    return WEXITSTATUS(result);
}

TEST_CASE("Error tests - missing and invalid")
{
    const std::string verif_exe_path = getVerifyFileLocation();
    const std::string refResultFile  = "ref.npy";
    const std::string impResultFile  = "imp.npy";

    SUBCASE("Test missing config file")
    {
        // Assumes json file is checked for first
        int exitCode = runVerifyCommand(verif_exe_path, "no-config.json", refResultFile, impResultFile);
        REQUIRE(exitCode == tvf_status_t::TVF_ERROR);
    }

    SUBCASE("Test invalid json in config file")
    {
        // Create a mock invalid JSON config file
        std::string invalidConfigFile = "invalid_config.json";
        createMockJsonFile(invalidConfigFile, "{invalid json}");

        std::vector<int32_t> shape  = { 3, 1 };
        std::vector<double> refData = { 5.1, 10.1, 3.0 };
        createMockNumpyFiles<double, float>(refResultFile, impResultFile, shape, refData);

        int exitCode = runVerifyCommand(verif_exe_path, invalidConfigFile, refResultFile, impResultFile);
        REQUIRE(exitCode == tvf_status_t::TVF_ERROR);

        // Clean up
        removeMockFile(invalidConfigFile);
        removeMockFile(refResultFile);
        removeMockFile(impResultFile);
    }

    SUBCASE("Test unsupported data type")
    {
        // Create a mock config file with unsupported data type
        std::string unsupportedConfigFile = "unsupported_data_type.json";
        createMockJsonFile(unsupportedConfigFile, R"({
            "ofm_name": [
                "output"
            ],
            "meta": {
                "compliance": {
                    "tensors": {
                        "output": {
                            "shape": [1, 3],
                            "data_type": "unsupported_type"
                        }
                    }
                }
            }
        })");

        std::vector<int32_t> shape  = { 1, 3 };
        std::vector<double> refData = { 5.1, 10.1, 3.0 };
        createMockNumpyFiles<double, float>(refResultFile, impResultFile, shape, refData);

        int exitCode = runVerifyCommand(verif_exe_path, unsupportedConfigFile, refResultFile, impResultFile);
        REQUIRE(exitCode == tvf_status_t::TVF_ERROR);

        // Clean up
        removeMockFile(unsupportedConfigFile);
        removeMockFile(refResultFile);
        removeMockFile(impResultFile);
    }

    SUBCASE("Test ofmName mismatch")
    {
        // Create a mock valid config file and result files
        std::string validConfigFile = "valid_config.json";

        std::vector<int32_t> shape  = { 3, 5 };
        std::vector<double> refData = { 5.1, 10.1, 3.0 };
        createMockNumpyFiles<double, float>(refResultFile, impResultFile, shape, refData);

        createMockJsonFile(validConfigFile, R"({
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
                            "shape": [3,5],
                            "ulp_info": {
                                "ulp": 0.5
                            }
                        }
                    }
                }
            }
        })");

        int exitCode =
            runVerifyCommand(verif_exe_path, validConfigFile, refResultFile, impResultFile, "invalid-tensor");
        REQUIRE(exitCode == tvf_status_t::TVF_ERROR);

        // Clean up
        removeMockFile(validConfigFile);
        removeMockFile(refResultFile);
        removeMockFile(impResultFile);
    }

    SUBCASE("Test multiple ofmName in config")
    {
        // Create a mock valid config file and result files
        std::string validConfigFile = "valid_config.json";

        std::vector<int32_t> shape  = { 5, 2 };
        std::vector<double> refData = { 5.1, 10.1, 3.0 };
        createMockNumpyFiles<double, float>(refResultFile, impResultFile, shape, refData);

        createMockJsonFile(validConfigFile, R"({
            "ofm_name": [
                "result-0",
                "result-1"
            ],
            "meta": {
                "compliance": {
                    "version": "0.1",
                    "tensors": {
                        "result-0": {
                            "mode": "ULP",
                            "data_type": "FP32",
                            "shape": [5,2],
                            "ulp_info": {
                                "ulp": 0.5
                            }
                        },
                        "result-1": {
                            "mode": "ULP",
                            "data_type": "FP32",
                            "shape": [5,2],
                            "ulp_info": {
                                "ulp": 0.5
                            }
                        }
                    }
                }
            }
        })");

        int exitCode = runVerifyCommand(verif_exe_path, validConfigFile, refResultFile, impResultFile);
        REQUIRE(exitCode == tvf_status_t::TVF_ERROR);

        // Clean up
        removeMockFile(validConfigFile);
        removeMockFile(refResultFile);
        removeMockFile(impResultFile);
    }

    SUBCASE("Test missing tensor data in implementation")
    {
        std::string validConfigFile = "valid_config.json";
        createMockJsonFile(validConfigFile, R"({
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
                            "shape": [5,2],
                            "ulp_info": {
                                "ulp": 0.5
                            }
                        }
                    }
                }
            }
        })");

        std::string missingResultFile = "missing.npy";

        std::vector<int32_t> shape  = { 5, 2 };
        std::vector<double> refData = { 5.1, 10.1, 3.0 };
        // Write both files, but just don't use one of them
        createMockNumpyFiles<double, float>(refResultFile, impResultFile, shape, refData);

        int exitCode = runVerifyCommand(verif_exe_path, validConfigFile, refResultFile, missingResultFile);
        REQUIRE(exitCode == tvf_status_t::TVF_ERROR);

        // Clean up
        removeMockFile(validConfigFile);
        removeMockFile(refResultFile);
        removeMockFile(impResultFile);
    }

    SUBCASE("Test missing tensor data in reference")
    {
        std::string validConfigFile = "valid_config.json";
        createMockJsonFile(validConfigFile, R"({
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
                            "shape": [5,2],
                            "ulp_info": {
                                "ulp": 0.5
                            }
                        }
                    }
                }
            }
        })");

        std::string missingResultFile = "missing.npy";

        std::vector<int32_t> shape  = { 5, 2 };
        std::vector<double> refData = { 5.1, 10.1, 3.0 };
        // Write both files, but just don't use one of them
        createMockNumpyFiles<double, float>(refResultFile, impResultFile, shape, refData);

        int exitCode = runVerifyCommand(verif_exe_path, validConfigFile, missingResultFile, impResultFile);
        REQUIRE(exitCode == tvf_status_t::TVF_ERROR);
        // Clean up
        removeMockFile(validConfigFile);
        removeMockFile(refResultFile);
        removeMockFile(impResultFile);
    }
}

TEST_CASE("Non-compliant tests")
{
    const std::string verif_exe_path = getVerifyFileLocation();
    const std::string refResultFile  = "ref.npy";
    const std::string impResultFile  = "imp.npy";

    SUBCASE("Test mismatching shapes")
    {
        // Create a mock config file with mismatching shapes
        std::string mismatchingShapesConfigFile = "mismatching_shapes.json";
        createMockJsonFile(mismatchingShapesConfigFile, R"({
            "ofm_name": [
                "output"
            ],
            "meta": {
                "compliance": {
                    "tensors": {
                        "output": {
                            "shape": [1, 2],
                            "data_type": "FP32"
                        }
                    }
                }
            }
        })");

        std::vector<int32_t> shape  = { 2, 1 };
        std::vector<double> refData = { 5.1, 10.1, 3.0 };
        createMockNumpyFiles<double, float>(refResultFile, impResultFile, shape, refData);

        int exitCode = runVerifyCommand(verif_exe_path, mismatchingShapesConfigFile, refResultFile, impResultFile);
        REQUIRE(exitCode == tvf_status_t::TVF_NON_COMPLIANT);

        // Clean up
        removeMockFile(mismatchingShapesConfigFile);
        removeMockFile(refResultFile);
        removeMockFile(impResultFile);
    }

    SUBCASE("Test mismatching shapes rank0 versus rank1")
    {
        // Create a mock config file with mismatching shapes
        std::string mismatchingShapesConfigFile = "mismatching_shapes.json";
        createMockJsonFile(mismatchingShapesConfigFile, R"({
            "ofm_name": [
                "output"
            ],
            "meta": {
                "compliance": {
                    "tensors": {
                        "output": {
                            "shape": [],
                            "data_type": "INT32"
                        }
                    }
                }
            }
        })");

        std::vector<int32_t> shape   = { 1 };
        std::vector<int32_t> refData = { 5 };
        createMockNumpyFiles<int32_t, int32_t>(refResultFile, impResultFile, shape, refData);

        int exitCode = runVerifyCommand(verif_exe_path, mismatchingShapesConfigFile, refResultFile, impResultFile);
        REQUIRE(exitCode == tvf_status_t::TVF_NON_COMPLIANT);

        // Clean up
        removeMockFile(mismatchingShapesConfigFile);
        removeMockFile(refResultFile);
        removeMockFile(impResultFile);
    }

    SUBCASE("Test mismatching shapes rank1 versus rank0")
    {
        // Create a mock config file with mismatching shapes
        std::string mismatchingShapesConfigFile = "mismatching_shapes.json";
        createMockJsonFile(mismatchingShapesConfigFile, R"({
            "ofm_name": [
                "output"
            ],
            "meta": {
                "compliance": {
                    "tensors": {
                        "output": {
                            "shape": [1],
                            "data_type": "INT32"
                        }
                    }
                }
            }
        })");

        std::vector<int32_t> shape   = {};
        std::vector<int32_t> refData = { 5 };
        createMockNumpyFiles<int32_t, int32_t>(refResultFile, impResultFile, shape, refData);

        int exitCode = runVerifyCommand(verif_exe_path, mismatchingShapesConfigFile, refResultFile, impResultFile);
        REQUIRE(exitCode == tvf_status_t::TVF_NON_COMPLIANT);

        // Clean up
        removeMockFile(mismatchingShapesConfigFile);
        removeMockFile(refResultFile);
        removeMockFile(impResultFile);
    }

    SUBCASE("Test mismatching datatypes")
    {
        // Create a mock config file with mismatching shapes
        std::string mismatchingTypesConfigFile = "mismatching_types.json";
        createMockJsonFile(mismatchingTypesConfigFile, R"({
            "ofm_name": [
                "output"
            ],
            "meta": {
                "compliance": {
                    "tensors": {
                        "output": {
                            "shape": [1,3,50],
                            "data_type": "INT32"
                        }
                    }
                }
            }
        })");

        std::vector<int32_t> shape  = { 1, 3, 50 };
        std::vector<double> refData = { 5.1 };
        createMockNumpyFiles<double, float>(refResultFile, impResultFile, shape, refData);

        int exitCode = runVerifyCommand(verif_exe_path, mismatchingTypesConfigFile, refResultFile, impResultFile);
        REQUIRE(exitCode == tvf_status_t::TVF_NON_COMPLIANT);

        // Clean up
        removeMockFile(mismatchingTypesConfigFile);
        removeMockFile(refResultFile);
        removeMockFile(impResultFile);
    }
}
TEST_CASE("Compliant tests")
{
    const std::string verif_exe_path = getVerifyFileLocation();
    const std::string refResultFile  = "ref.npy";
    const std::string impResultFile  = "imp.npy";

    SUBCASE("Test correct usage of ofmName")
    {
        // Create a mock valid config file and result files
        std::string validConfigFile = "valid_config.json";
        createMockJsonFile(validConfigFile, R"({
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

        std::vector<int32_t> shape  = { 35, 32 };
        std::vector<double> refData = { 5.1, 10.1, 3.0 };
        createMockNumpyFiles<double, float>(refResultFile, impResultFile, shape, refData);

        int exitCode = runVerifyCommand(verif_exe_path, validConfigFile, refResultFile, impResultFile, "result-0");
        REQUIRE(exitCode == tvf_status_t::TVF_COMPLIANT);

        // Clean up
        removeMockFile(validConfigFile);
        removeMockFile(refResultFile);
        removeMockFile(impResultFile);
    }

    SUBCASE("Test fp32")
    {
        // Create a mock valid config file and result files
        std::string validConfigFile = "valid_config.json";
        createMockJsonFile(validConfigFile, R"({
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

        std::vector<int32_t> shape  = { 35, 32 };
        std::vector<double> refData = { 5.1, 10.1, 3.0 };
        createMockNumpyFiles<double, float>(refResultFile, impResultFile, shape, refData);

        int exitCode = runVerifyCommand(verif_exe_path, validConfigFile, refResultFile, impResultFile);
        REQUIRE(exitCode == tvf_status_t::TVF_COMPLIANT);

        // Clean up
        removeMockFile(validConfigFile);
        removeMockFile(refResultFile);
        removeMockFile(impResultFile);
    }

    SUBCASE("Test int32")
    {
        // Create a mock valid config file and result files dynamically
        std::string validConfigFile = "valid_config_int32.json";
        createMockJsonFile(validConfigFile, R"({
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

        std::vector<int32_t> shape   = { 2, 12, 10, 7 };
        std::vector<int32_t> refData = { 5, 10, 3 };
        createMockNumpyFiles<int32_t, int32_t>(refResultFile, impResultFile, shape, refData);

        int exitCode = runVerifyCommand(verif_exe_path, validConfigFile, refResultFile, impResultFile);
        REQUIRE(exitCode == tvf_status_t::TVF_COMPLIANT);

        // Clean up dynamically generated files
        removeMockFile(validConfigFile);
        removeMockFile(refResultFile);
        removeMockFile(impResultFile);
    }
}
#endif    // __unix__
TEST_SUITE_END();

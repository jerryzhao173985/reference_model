
// Copyright (c) 2024, ARM Limited.
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

#include "cxxopts.hpp"
#include "func_debug.h"
#include "generate_utils.cc"
#include "tensor.h"
#include "verify_entry.cc"
#include <cstring>
#include <dtype.h>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdio.h>
#include <string>
#include <types.h>

using json      = nlohmann::json;
using EigenType = typename TosaReference::GetEigenType<TosaReference::TOSA_REF_TYPE_SHAPE>::type;
using TOut      = Eigen::Tensor<EigenType, 1>;

int initTestDesc(json& test_desc, const char* desc_path)
{
    std::ifstream ifs(desc_path);

    if (ifs.good())
    {
        try
        {
            test_desc = nlohmann::json::parse(ifs);
            if (!test_desc.contains("meta") || !test_desc["meta"].contains("compliance") ||
                !test_desc["meta"]["compliance"].contains("tensors"))
            {
                WARNING("Invalid json test descriptor - missing information (meta/compliance/tensors) required for "
                        "validation.");
                return 1;
            }
        }
        catch (nlohmann::json::parse_error& e)
        {
            WARNING("Error parsing test descriptor json: %s", e.what());
            return 1;
        }
    }
    else
    {
        WARNING("Error opening json test descriptor - %s", desc_path);
        return 1;
    }
    return 0;
}

// Read the command line arguments
int parse_cmd_line(int argc,
                   char** argv,
                   std::string* configFile,
                   std::string* refFile,
                   std::string* impFile,
                   std::string* bndFile,
                   std::string* ofmName)
{
    try
    {
        cxxopts::Options options("Verify", "The verify executable");

        // clang-format off
        options.add_options()
        ("test_desc", "(Required) Json test descriptor", cxxopts::value<std::string>(*configFile), "<descriptor>")
        ("ofm_name", "name of the output tensor to check, defaults to the first ofm_name listed in the test", cxxopts::value<std::string>(*ofmName))
        ("imp_result_file", "(Required) path to the implementation result file to check", cxxopts::value<std::string>(*impFile))
        ("ref_result_file", "(Required) path to the reference model result file to check", cxxopts::value<std::string>(*refFile))
        ("bnd_result_file", "path to the reference model bounds result file produced by certain tests for compliance checking", cxxopts::value<std::string>(*bndFile))
        ("h,help", "print help");
        // clang-format on

        auto result = options.parse(argc, argv);
        if (result.count("help") || !result.count("test_desc") || !result.count("imp_result_file") ||
            !result.count("ref_result_file"))
        {
            std::cout << options.help() << std::endl;
            return 1;
        }
    }
    catch (const std::exception& e)
    {
        WARNING(e.what());
        return 1;
    }

    return 0;
}

DType mapToDType(const std::string dataType)
{
    static std::map<std::string, DType> typeMap = {
        { "BOOL", DType_BOOL },   { "INT4", DType_INT4 },   { "INT8", DType_INT8 },       { "UINT16", DType_UINT16 },
        { "INT16", DType_INT16 }, { "INT32", DType_INT32 }, { "INT48", DType_INT48 },     { "FP16", DType_FP16 },
        { "BF16", DType_BF16 },   { "FP32", DType_FP32 },   { "FP8E4M3", DType_FP8E4M3 }, { "FP8E5M2", DType_FP8E5M2 },
        { "UINT8", DType_UINT8 },
    };

    if (typeMap.count(dataType))
    {
        return typeMap[dataType];
    }

    return DType_UNKNOWN;
}

tosa_datatype_t mapToTosaDtype(const DType dataType, const bool isFP64)
{
    if ((dataType == DType_FP16 || dataType == DType_BF16 || dataType == DType_FP32 || dataType == DType_FP8E4M3 ||
         dataType == DType_FP8E5M2) &&
        isFP64)
    {
        return tosa_datatype_fp64_t;
    }

    static std::map<DType, tosa_datatype_t> typeMap = {
        { DType_BOOL, tosa_datatype_bool_t },
        { DType_INT4, tosa_datatype_int4_t },
        { DType_INT8, tosa_datatype_int8_t },
        { DType_UINT16, tosa_datatype_uint16_t },
        { DType_INT16, tosa_datatype_int16_t },
        { DType_INT32, tosa_datatype_int32_t },
        { DType_INT48, tosa_datatype_int48_t },
        { DType_FP16, tosa_datatype_fp16_t },
        { DType_BF16, tosa_datatype_bf16_t },
        {
            DType_FP32,
            tosa_datatype_fp32_t,
        },
        { DType_SHAPE, tosa_datatype_shape_t },
        {
            DType_FP8E4M3,
            tosa_datatype_fp8e4m3_t,
        },
        { DType_FP8E5M2, tosa_datatype_fp8e5m2_t },
        { DType_UINT8, tosa_datatype_uint8_t },
    };

    // if (typeMap.count(dataType))
    // {
    return typeMap[dataType];
    // }

    //return uint8 in the case of failure
    //return tosa_datatype_uint8_t;
    // WARNING("Unsupported datatype");
    // return tosa;
}

//loads numpy file and put it into the tosa_tensor struct for usage in verify library
int createTensorMap(const char* resultFile,
                    tosa_tensor_t* resultMapTensor,
                    const DType dtype,
                    const char* ofmName,
                    const std::vector<int32_t> shape,
                    const int elems)
{

    // Pointer to the buffer that will be used in tosa_tensor_t
    uint8_t* databuf = nullptr;

    bool success = false;

    if (dtype == DType_BOOL)
    {
        bool* bdatabuf                  = new bool[elems];
        NumpyUtilities::NPError nperror = NumpyUtilities::readFromNpyFile(resultFile, elems, bdatabuf);
        if (nperror == NumpyUtilities::NO_ERROR)
        {
            databuf = reinterpret_cast<uint8_t*>(bdatabuf);
            success = true;
        }
        else
        {
            delete[] bdatabuf;
        }
    }
    else if ((dtype == DType_FP16 || dtype == DType_FP32 || dtype == DType_FP8E4M3 || dtype == DType_FP8E5M2 ||
              dtype == DType_BF16) &&
             g_func_config.precise_mode)
    {
        double* doubleDataBuf           = new double[elems];
        NumpyUtilities::NPError nperror = NumpyUtilities::readFromNpyFile(resultFile, elems, doubleDataBuf);
        if (nperror == NumpyUtilities::NO_ERROR)
        {
            databuf = reinterpret_cast<uint8_t*>(doubleDataBuf);
            success = true;
        }
        else
        {
            delete[] doubleDataBuf;
        }
    }
    else if (dtype == DType_BF16)
    {
        bf16* bf16dataBuf               = new bf16[elems];
        NumpyUtilities::NPError nperror = NumpyUtilities::readFromNpyFile(resultFile, elems, bf16dataBuf);
        if (nperror == NumpyUtilities::NO_ERROR)
        {
            databuf = reinterpret_cast<uint8_t*>(bf16dataBuf);
            success = true;
        }
        else
        {
            delete[] bf16dataBuf;
        }
    }
    else if (dtype == DType_FP16)
    {
        half_float::half* fp16dataBuf   = new half_float::half[elems];
        NumpyUtilities::NPError nperror = NumpyUtilities::readFromNpyFile(resultFile, elems, fp16dataBuf);
        if (nperror == NumpyUtilities::NO_ERROR)
        {
            databuf = reinterpret_cast<uint8_t*>(fp16dataBuf);
            success = true;
        }
        else
        {
            delete[] fp16dataBuf;
        }
    }
    else if (dtype == DType_FP32)
    {
        float* fp32dataBuf              = new float[elems];
        NumpyUtilities::NPError nperror = NumpyUtilities::readFromNpyFile(resultFile, elems, fp32dataBuf);
        if (nperror == NumpyUtilities::NO_ERROR)
        {
            databuf = reinterpret_cast<uint8_t*>(fp32dataBuf);
            success = true;
        }
        else
        {
            delete[] fp32dataBuf;
        }
    }
    else if (dtype == DType_FP8E4M3)
    {
        fp8e4m3* fp8e4m3DataBuf         = new fp8e4m3[elems];
        NumpyUtilities::NPError nperror = NumpyUtilities::readFromNpyFile(resultFile, elems, fp8e4m3DataBuf);
        if (nperror == NumpyUtilities::NO_ERROR)
        {
            databuf = reinterpret_cast<uint8_t*>(fp8e4m3DataBuf);
            success = true;
        }
        else
        {
            delete[] fp8e4m3DataBuf;
        }
    }
    else if (dtype == DType_FP8E5M2)
    {
        fp8e5m2* fp8e5m2DataBuf         = new fp8e5m2[elems];
        NumpyUtilities::NPError nperror = NumpyUtilities::readFromNpyFile(resultFile, elems, fp8e5m2DataBuf);
        if (nperror == NumpyUtilities::NO_ERROR)
        {
            databuf = reinterpret_cast<uint8_t*>(fp8e5m2DataBuf);
            success = true;
        }
        else
        {
            delete[] fp8e5m2DataBuf;
        }
    }
    else if (dtype == DType_INT16)
    {
        int16_t* i16databuf             = new int16_t[elems];
        NumpyUtilities::NPError nperror = NumpyUtilities::readFromNpyFile(resultFile, elems, i16databuf);
        if (nperror == NumpyUtilities::NO_ERROR)
        {
            databuf = reinterpret_cast<uint8_t*>(i16databuf);
            success = true;
        }
        else
        {
            delete[] i16databuf;
        }
    }
    else if (dtype == DType_INT48)
    {
        int64_t* i64databuf             = new int64_t[elems];
        NumpyUtilities::NPError nperror = NumpyUtilities::readFromNpyFile(resultFile, elems, i64databuf);
        if (nperror == NumpyUtilities::NO_ERROR)
        {
            databuf = reinterpret_cast<uint8_t*>(i64databuf);
            success = true;
        }
        else
        {
            delete[] i64databuf;
        }
    }
    else if (dtype == DType_UINT8)
    {
        uint8_t* u8databuf              = new uint8_t[elems];
        NumpyUtilities::NPError nperror = NumpyUtilities::readFromNpyFile(resultFile, elems, u8databuf);
        if (nperror == NumpyUtilities::NO_ERROR)
        {
            databuf = u8databuf;
            success = true;
        }
        else
        {
            delete[] u8databuf;
        }
    }
    else if (dtype == DType_UINT16)
    {
        uint16_t* u16databuf            = new uint16_t[elems];
        NumpyUtilities::NPError nperror = NumpyUtilities::readFromNpyFile(resultFile, elems, u16databuf);
        if (nperror == NumpyUtilities::NO_ERROR)
        {
            databuf = reinterpret_cast<uint8_t*>(u16databuf);
            success = true;
        }
        else
        {
            delete[] u16databuf;
        }
    }
    else if (dtype == DType_INT8)
    {
        int8_t* i8databuf               = new int8_t[elems];
        NumpyUtilities::NPError nperror = NumpyUtilities::readFromNpyFile(resultFile, elems, i8databuf);
        if (nperror == NumpyUtilities::NO_ERROR)
        {
            databuf = reinterpret_cast<uint8_t*>(i8databuf);
            success = true;
        }
        else
        {
            delete[] i8databuf;
        }
    }
    else if (dtype == DType_INT32)
    {
        int32_t* i32databuf             = new int32_t[elems];
        NumpyUtilities::NPError nperror = NumpyUtilities::readFromNpyFile(resultFile, elems, i32databuf);
        if (nperror == NumpyUtilities::NO_ERROR)
        {
            databuf = reinterpret_cast<uint8_t*>(i32databuf);
            success = true;
        }
        else
        {
            delete[] i32databuf;
        }
    }
    else
    {
        WARNING("Unsupported data type!");
    }
    // If we failed to read the tensor data, clean up and return error
    if (!success)
    {
        WARNING("Failed to get buffer from %s", resultFile);

        return 1;
    }

    // Populate the tosa_tensor_t structure
    resultMapTensor->data = databuf;

    return 0;
}

int main(int argc, char* argv[])
{

    std::string configFile;
    std::string impResultFile;
    std::string refResultFile;
    std::string bndResultFile;
    std::string ofmName;

    if (parse_cmd_line(argc, argv, &configFile, &refResultFile, &impResultFile, &bndResultFile, &ofmName))
    {

        return 1;
    }

    json test_desc;

    // Initialize test descriptor
    if (initTestDesc(test_desc, configFile.c_str()))
    {
        // Errors will be reported by the initTestDesc function
        return 1;
    }
    if (ofmName.empty())
    {
        if (test_desc["ofm_name"].size() > 1)
        {
            WARNING("More than one test output file, please specify which one should be validated using the "
                    "`--ofm_name` option.");
            return 1;
        }
        ofmName = (test_desc["ofm_name"][0]).get<std::string>();
    }

    if (!test_desc["meta"]["compliance"]["tensors"].contains(ofmName))
    {
        WARNING(("Invalid json test descriptor - missing information about ofmName - " + ofmName).c_str());
        return 1;
    }

    if (!test_desc["meta"]["compliance"]["tensors"][ofmName].contains("shape"))
    {
        WARNING("Invalid json test descriptor - missing shape information required for validation.");
        return 1;
    }

    std::vector<int> shape = test_desc["meta"]["compliance"]["tensors"][ofmName]["shape"];

    const auto elems = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<int64_t>());

    if (!test_desc["meta"]["compliance"]["tensors"][ofmName].contains("data_type"))
    {
        WARNING("Invalid json test descriptor - missing datatype information required for validation.");
        return 1;
    }

    std::string typeText = test_desc["meta"]["compliance"]["tensors"][ofmName]["data_type"];

    DType dtype = mapToDType(typeText);

    if (dtype == DType_UNKNOWN)
    {
        WARNING("Unsupported data type in json test descriptor");
        return 1;
    }

    //implementation result numpy into a data buffer

    tosa_tensor_t* imp = new tosa_tensor_t;

    if (createTensorMap(impResultFile.c_str(), imp, dtype, ofmName.c_str(), shape, elems))
    {
        delete imp;

        WARNING("Failed to create implementation tensor type");
        return 1;
    }

    imp->name      = ofmName.c_str();
    imp->data_type = mapToTosaDtype(dtype, false);
    imp->num_dims  = static_cast<int32_t>(shape.size());
    imp->size      = TosaReference::elementSizeFromType(dtype);
    imp->shape     = shape.data();

    tosa_tensor_t* ref = new tosa_tensor_t;

    //where floating point implementation is used, fp64 is used for reference
    //so turning precise mode allows for support of fp64
    g_func_config.precise_mode = true;

    if (createTensorMap(refResultFile.c_str(), ref, dtype, ofmName.c_str(), shape, elems))
    {

        delete ref;
        WARNING("Failed to create reference tensor type");
        return 1;
    }

    ref->name      = ofmName.c_str();
    ref->data_type = mapToTosaDtype(dtype, true);
    ref->num_dims  = static_cast<int32_t>(shape.size());
    ref->size      = TosaReference::elementSizeFromType(dtype);
    ref->shape     = shape.data();

    tosa_tensor_t* bnd = nullptr;

    if (!bndResultFile.empty())
    {

        bnd = new tosa_tensor_t;
        if (createTensorMap(bndResultFile.c_str(), bnd, dtype, ofmName.c_str(), shape, elems))
        {
            delete bnd;
            WARNING("Failed to create bound tensor type");
        }

        bnd->name      = ofmName.c_str();
        bnd->data_type = mapToTosaDtype(dtype, true);
        bnd->num_dims  = static_cast<int32_t>(shape.size());
        bnd->size      = TosaReference::elementSizeFromType(dtype);
        bnd->shape     = shape.data();
    }

    g_func_config.precise_mode = false;

    // Serialize the JSON object to a string
    std::string testDescString = test_desc["meta"]["compliance"].dump();
    bool success               = true;
    if (tvf_verify_data(ref, bnd, imp, testDescString.c_str()))
    {
        std::cout << "Test passed" << std::endl;
    }
    else
    {
        WARNING("Test Failed");
        success = false;
    }

    if (imp)
    {
        delete[] imp->data;
        delete imp;
    }
    if (ref)
    {
        delete[] ref->data;
        delete ref;
    }
    if (bnd)
    {
        delete[] bnd->data;
        delete bnd;
    }
    if (!success)
    {
        return 1;
    }
    return 0;
}

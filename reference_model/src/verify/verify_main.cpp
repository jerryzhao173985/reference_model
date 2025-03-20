
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

#include "cxxopts.hpp"
#include "dtype.h"
#include "func_debug.h"
#include "generate_utils.h"
#include "numpy_utils.h"
#include "verifiers.h"
#include "verify.h"
#include "verify_utils.h"
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

bool initTestDesc(json& test_desc, const char* desc_path)
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
                WARNING("[Verifier] Invalid json test descriptor - missing information (meta/compliance/tensors) "
                        "required for validation.");
                return false;
            }
        }
        catch (nlohmann::json::parse_error& e)
        {
            WARNING("[Verifier] Error parsing test descriptor json: %s", e.what());
            return false;
        }
    }
    else
    {
        WARNING("[Verifier] Error opening json test descriptor - %s", desc_path);
        return false;
    }
    return true;
}

// Read the command line arguments
bool parseCmdLine(int argc,
                  char** argv,
                  std::string* configFile,
                  std::string* refFile,
                  std::string* impFile,
                  std::string* bndFile,
                  std::string* ofmName)
{
    try
    {
        cxxopts::Options options("tosa_verify", "The TOSA test result verifier");

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
        if (result.count("help"))
        {
            std::cout << options.help() << std::endl;
            return false;
        }
        if (!result.count("test_desc") || !result.count("imp_result_file") || !result.count("ref_result_file"))
        {
            WARNING(
                "[Verifier] Missing one or more required arguments: --test_desc, --imp_result_file, --ref_result_file");
            return false;
        }
    }
    catch (const std::exception& e)
    {
        WARNING("[Verifier] %s", e.what());
        return false;
    }

    return true;
}

DType mapToDType(const std::string dataType)
{
    static std::map<std::string, DType> typeMap = {
        { "BOOL", DType_BOOL },   { "INT4", DType_INT4 },   { "INT8", DType_INT8 },       { "UINT16", DType_UINT16 },
        { "INT16", DType_INT16 }, { "INT32", DType_INT32 }, { "INT48", DType_INT48 },     { "FP16", DType_FP16 },
        { "BF16", DType_BF16 },   { "FP32", DType_FP32 },   { "FP8E4M3", DType_FP8E4M3 }, { "FP8E5M2", DType_FP8E5M2 },
        { "UINT8", DType_UINT8 }, { "SHAPE", DType_SHAPE },
    };

    if (typeMap.count(dataType))
    {
        return typeMap[dataType];
    }

    return DType_UNKNOWN;
}

tosa_datatype_t mapToTosaDataType(const TosaReference::TOSA_REF_TYPE tosaRefType)
{
    static std::map<TosaReference::TOSA_REF_TYPE, tosa_datatype_t> typeMap = {
        { TosaReference::TOSA_REF_TYPE_BOOL, tosa_datatype_bool_t },
        { TosaReference::TOSA_REF_TYPE_INT4, tosa_datatype_int4_t },
        { TosaReference::TOSA_REF_TYPE_INT8, tosa_datatype_int8_t },
        { TosaReference::TOSA_REF_TYPE_INT16, tosa_datatype_int16_t },
        { TosaReference::TOSA_REF_TYPE_INT32, tosa_datatype_int32_t },
        { TosaReference::TOSA_REF_TYPE_INT48, tosa_datatype_int48_t },
        { TosaReference::TOSA_REF_TYPE_SHAPE, tosa_datatype_shape_t },
        { TosaReference::TOSA_REF_TYPE_FP8E4M3, tosa_datatype_fp8e4m3_t },
        { TosaReference::TOSA_REF_TYPE_FP8E5M2, tosa_datatype_fp8e5m2_t },
        { TosaReference::TOSA_REF_TYPE_FP16, tosa_datatype_fp16_t },
        { TosaReference::TOSA_REF_TYPE_BF16, tosa_datatype_bf16_t },
        { TosaReference::TOSA_REF_TYPE_FP32, tosa_datatype_fp32_t },
        { TosaReference::TOSA_REF_TYPE_FP64, tosa_datatype_fp64_t },
    };

    return typeMap[tosaRefType];
}

tvf_status_t validateResultFile(const std::string resultFile,
                                const TosaReference::TOSA_REF_TYPE tosaRefType,
                                const std::vector<int32_t>& shape)
{
    NumpyUtilities::NPError nperror;

    switch (tosaRefType)
    {
        case TosaReference::TOSA_REF_TYPE_BOOL: {
            nperror = NumpyUtilities::validateNpyHeader<bool>(resultFile.c_str(), shape);
            break;
        }
        case TosaReference::TOSA_REF_TYPE_INT4:
            [[fallthrough]];
        case TosaReference::TOSA_REF_TYPE_INT8: {
            nperror = NumpyUtilities::validateNpyHeader<int8_t>(resultFile.c_str(), shape);
            break;
        }
        case TosaReference::TOSA_REF_TYPE_INT16: {
            nperror = NumpyUtilities::validateNpyHeader<int16_t>(resultFile.c_str(), shape);
            break;
        }
        case TosaReference::TOSA_REF_TYPE_INT32: {
            nperror = NumpyUtilities::validateNpyHeader<int32_t>(resultFile.c_str(), shape);
            break;
        }
        case TosaReference::TOSA_REF_TYPE_INT48:
            [[fallthrough]];
        case TosaReference::TOSA_REF_TYPE_SHAPE: {
            nperror = NumpyUtilities::validateNpyHeader<int64_t>(resultFile.c_str(), shape);
            break;
        }
        case TosaReference::TOSA_REF_TYPE_FP8E4M3: {
            nperror = NumpyUtilities::validateNpyHeader<fp8e4m3>(resultFile.c_str(), shape);
            break;
        }
        case TosaReference::TOSA_REF_TYPE_FP8E5M2: {
            nperror = NumpyUtilities::validateNpyHeader<fp8e5m2>(resultFile.c_str(), shape);
            break;
        }
        case TosaReference::TOSA_REF_TYPE_FP16: {
            nperror = NumpyUtilities::validateNpyHeader<half_float::half>(resultFile.c_str(), shape);
            break;
        }
        case TosaReference::TOSA_REF_TYPE_BF16: {
            nperror = NumpyUtilities::validateNpyHeader<bf16>(resultFile.c_str(), shape);
            break;
        }
        case TosaReference::TOSA_REF_TYPE_FP32: {
            nperror = NumpyUtilities::validateNpyHeader<float>(resultFile.c_str(), shape);
            break;
        }
        case TosaReference::TOSA_REF_TYPE_FP64: {
            nperror = NumpyUtilities::validateNpyHeader<double>(resultFile.c_str(), shape);
            break;
        }
        default: {
            WARNING("[Verifier] Unsupported Numpy data type %s", TosaReference::EnumNameTOSAREFTYPE(tosaRefType));
            return tvf_status_t::TVF_ERROR;
        }
    }
    if (nperror == NumpyUtilities::NO_ERROR)
    {
        return tvf_status_t::TVF_COMPLIANT;
    }
    else if (nperror == NumpyUtilities::SHAPE_MISMATCH)
    {
        WARNING("[Verifier] Shape does not match %s in %s", TosaReference::positionToString(shape).c_str(),
                resultFile.c_str());
        return tvf_status_t::TVF_NON_COMPLIANT;
    }
    else if (nperror == NumpyUtilities::FILE_TYPE_MISMATCH)
    {
        WARNING("[Verifier] Data type/format not suitable for %s in %s",
                TosaReference::EnumNameTOSAREFTYPE(tosaRefType), resultFile.c_str());
        return tvf_status_t::TVF_NON_COMPLIANT;
    }
    else
    {
        WARNING("[Verifier] Invalid Numpy file %s - %s", resultFile.c_str(), NumpyUtilities::getErrorString(nperror));
        return tvf_status_t::TVF_ERROR;
    }
}

// Loads numpy file and put it into the tosa_tensor struct for usage in verify library
bool createTensorMap(const std::string resultFile,
                     const TosaReference::TOSA_REF_TYPE tosaRefType,
                     const std::string& tensorName,
                     std::vector<int32_t>& shape,
                     tosa_tensor_t* resultMapTensor)
{
    ASSERT_MEM(resultMapTensor);

    const int64_t elements = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<int64_t>());

    uint8_t* databuf                = nullptr;    // Pointer to the buffer that will be used in tosa_tensor_t
    size_t databuf_size             = 0;
    NumpyUtilities::NPError nperror = NumpyUtilities::NO_ERROR;

    // Set up minimal empty map for failure
    resultMapTensor->data = databuf;
    resultMapTensor->size = databuf_size;

    if (tosaRefType == TosaReference::TOSA_REF_TYPE_BOOL)
    {
        bool* bdatabuf = new bool[elements];
        nperror        = NumpyUtilities::readFromNpyFile(resultFile.c_str(), elements, bdatabuf);
        if (nperror == NumpyUtilities::NO_ERROR)
        {
            databuf_size = sizeof(*bdatabuf) * elements;
            databuf      = reinterpret_cast<uint8_t*>(bdatabuf);
        }
        else
        {
            delete[] bdatabuf;
        }
    }
    else if (tosaRefType == TosaReference::TOSA_REF_TYPE_INT4 || tosaRefType == TosaReference::TOSA_REF_TYPE_INT8)
    {
        // Numpy stores a single INT4 value in an int8 as it doesn't support
        // the packed format
        int8_t* i8databuf = new int8_t[elements];
        nperror           = NumpyUtilities::readFromNpyFile(resultFile.c_str(), elements, i8databuf);
        if (nperror == NumpyUtilities::NO_ERROR)
        {
            databuf_size = sizeof(*i8databuf) * elements;
            databuf      = reinterpret_cast<uint8_t*>(i8databuf);
        }
        else
        {
            delete[] i8databuf;
        }
    }
    else if (tosaRefType == TosaReference::TOSA_REF_TYPE_INT16)
    {
        int16_t* i16databuf = new int16_t[elements];
        nperror             = NumpyUtilities::readFromNpyFile(resultFile.c_str(), elements, i16databuf);
        if (nperror == NumpyUtilities::NO_ERROR)
        {
            databuf_size = sizeof(*i16databuf) * elements;
            databuf      = reinterpret_cast<uint8_t*>(i16databuf);
        }
        else
        {
            delete[] i16databuf;
        }
    }
    else if (tosaRefType == TosaReference::TOSA_REF_TYPE_INT32)
    {
        int32_t* i32databuf = new int32_t[elements];
        nperror             = NumpyUtilities::readFromNpyFile(resultFile.c_str(), elements, i32databuf);
        if (nperror == NumpyUtilities::NO_ERROR)
        {
            databuf_size = sizeof(*i32databuf) * elements;
            databuf      = reinterpret_cast<uint8_t*>(i32databuf);
        }
        else
        {
            delete[] i32databuf;
        }
    }
    else if (tosaRefType == TosaReference::TOSA_REF_TYPE_INT48 || tosaRefType == TosaReference::TOSA_REF_TYPE_SHAPE)
    {
        // Numpy stores INT48 values in an int64 as it doesn't support
        // the packed format
        int64_t* i64databuf = new int64_t[elements];
        nperror             = NumpyUtilities::readFromNpyFile(resultFile.c_str(), elements, i64databuf);
        if (nperror == NumpyUtilities::NO_ERROR)
        {
            databuf_size = sizeof(*i64databuf) * elements;
            databuf      = reinterpret_cast<uint8_t*>(i64databuf);
        }
        else
        {
            delete[] i64databuf;
        }
    }
    else if (tosaRefType == TosaReference::TOSA_REF_TYPE_FP8E4M3)
    {
        fp8e4m3* fp8e4m3DataBuf = new fp8e4m3[elements];
        nperror                 = NumpyUtilities::readFromNpyFile(resultFile.c_str(), elements, fp8e4m3DataBuf);
        if (nperror == NumpyUtilities::NO_ERROR)
        {
            databuf_size = sizeof(*fp8e4m3DataBuf) * elements;
            databuf      = reinterpret_cast<uint8_t*>(fp8e4m3DataBuf);
        }
        else
        {
            delete[] fp8e4m3DataBuf;
        }
    }
    else if (tosaRefType == TosaReference::TOSA_REF_TYPE_FP8E5M2)
    {
        fp8e5m2* fp8e5m2DataBuf = new fp8e5m2[elements];
        nperror                 = NumpyUtilities::readFromNpyFile(resultFile.c_str(), elements, fp8e5m2DataBuf);
        if (nperror == NumpyUtilities::NO_ERROR)
        {
            databuf_size = sizeof(*fp8e5m2DataBuf) * elements;
            databuf      = reinterpret_cast<uint8_t*>(fp8e5m2DataBuf);
        }
        else
        {
            delete[] fp8e5m2DataBuf;
        }
    }
    else if (tosaRefType == TosaReference::TOSA_REF_TYPE_FP16)
    {
        half_float::half* fp16dataBuf = new half_float::half[elements];
        nperror                       = NumpyUtilities::readFromNpyFile(resultFile.c_str(), elements, fp16dataBuf);
        if (nperror == NumpyUtilities::NO_ERROR)
        {
            databuf_size = sizeof(*fp16dataBuf) * elements;
            databuf      = reinterpret_cast<uint8_t*>(fp16dataBuf);
        }
        else
        {
            delete[] fp16dataBuf;
        }
    }
    else if (tosaRefType == TosaReference::TOSA_REF_TYPE_BF16)
    {
        bf16* bf16dataBuf = new bf16[elements];
        nperror           = NumpyUtilities::readFromNpyFile(resultFile.c_str(), elements, bf16dataBuf);
        if (nperror == NumpyUtilities::NO_ERROR)
        {
            databuf_size = sizeof(*bf16dataBuf) * elements;
            databuf      = reinterpret_cast<uint8_t*>(bf16dataBuf);
        }
        else
        {
            delete[] bf16dataBuf;
        }
    }
    else if (tosaRefType == TosaReference::TOSA_REF_TYPE_FP32)
    {
        float* fp32dataBuf = new float[elements];
        nperror            = NumpyUtilities::readFromNpyFile(resultFile.c_str(), elements, fp32dataBuf);
        if (nperror == NumpyUtilities::NO_ERROR)
        {
            databuf_size = sizeof(*fp32dataBuf) * elements;
            databuf      = reinterpret_cast<uint8_t*>(fp32dataBuf);
        }
        else
        {
            delete[] fp32dataBuf;
        }
    }
    else if (tosaRefType == TosaReference::TOSA_REF_TYPE_FP64)
    {
        double* doubleDataBuf = new double[elements];
        nperror               = NumpyUtilities::readFromNpyFile(resultFile.c_str(), elements, doubleDataBuf);
        if (nperror == NumpyUtilities::NO_ERROR)
        {
            databuf_size = sizeof(*doubleDataBuf) * elements;
            databuf      = reinterpret_cast<uint8_t*>(doubleDataBuf);
        }
        else
        {
            delete[] doubleDataBuf;
        }
    }
    else
    {
        // validateResultFile should have already checked this
        WARNING("[Verifier] INTERNAL ERROR: Unsupported Numpy data type");
        return false;
    }

    // If we failed to read the tensor data, clean up and return error
    if (nperror != NumpyUtilities::NO_ERROR)
    {
        WARNING("[Verifier] Failed to load data from %s - %s", resultFile.c_str(),
                NumpyUtilities::getErrorString(nperror));
        return false;
    }

    // Populate the tosa_tensor_t structure
    resultMapTensor->name      = tensorName.c_str();
    resultMapTensor->data      = databuf;
    resultMapTensor->size      = databuf_size;
    resultMapTensor->data_type = mapToTosaDataType(tosaRefType);
    resultMapTensor->num_dims  = static_cast<int32_t>(shape.size());
    resultMapTensor->shape     = shape.data();

    return true;
}

int main(int argc, char* argv[])
{
    std::string configFile;
    std::string impResultFile;
    std::string refResultFile;
    std::string bndResultFile;
    std::string ofmName;

    if (!parseCmdLine(argc, argv, &configFile, &refResultFile, &impResultFile, &bndResultFile, &ofmName))
    {
        return tvf_status_t::TVF_ERROR;
    }

    json test_desc;

    // Initialize test descriptor
    if (!initTestDesc(test_desc, configFile.c_str()))
    {
        // Errors will be reported by the initTestDesc function
        return tvf_status_t::TVF_ERROR;
    }
    if (ofmName.empty())
    {
        if (test_desc["ofm_name"].size() > 1)
        {
            WARNING("[Verifier] More than one test output file, please specify which one should be validated using the "
                    "`--ofm_name` option.");
            return tvf_status_t::TVF_ERROR;
        }
        ofmName = (test_desc["ofm_name"][0]).get<std::string>();
    }

    if (!test_desc["meta"]["compliance"]["tensors"].contains(ofmName))
    {
        WARNING("Invalid json test descriptor - missing tensor information for ofm_name: %s", ofmName.c_str());
        return tvf_status_t::TVF_ERROR;
    }

    if (!test_desc["meta"]["compliance"]["tensors"][ofmName].contains("shape"))
    {
        WARNING("[Verifier] Invalid json test descriptor - missing shape information required for validation.");
        return tvf_status_t::TVF_ERROR;
    }

    std::vector<int32_t> shape = test_desc["meta"]["compliance"]["tensors"][ofmName]["shape"];
    if (!test_desc["meta"]["compliance"]["tensors"][ofmName].contains("data_type"))
    {
        WARNING("[Verifier] Invalid json test descriptor - missing datatype information required for validation.");
        return tvf_status_t::TVF_ERROR;
    }

    std::string typeText = test_desc["meta"]["compliance"]["tensors"][ofmName]["data_type"];

    DType dtype = mapToDType(typeText);

    if (dtype == DType_UNKNOWN)
    {
        WARNING("[Verifier] Invalid json test descriptor - unsupported data type: %s", typeText.c_str());
        return tvf_status_t::TVF_ERROR;
    }

    // NOTE: we are likely to change the representation of the shape of a rank 0 tensor in the future
    // TODO(ITL) if we don't change the way we represent shapes of rank 0 tensors, implement this as
    // a utility in the serialization library instead
    const std::vector<int32_t> shapeOfRank0Shape = { 0 };
    std::vector<int32_t> npyShape = (dtype == DType_SHAPE) && (shape.size() == 0) ? shapeOfRank0Shape : shape;

    // Convert type to TOSA_REF_TYPE
    TosaReference::TOSA_REF_TYPE impTosaRefType = TosaReference::ConvertDType(dtype, false);

    // For floating point results, fp64 is used for reference data
    // Set precise mode to indicate this
    TosaReference::TOSA_REF_TYPE refTosaRefType = TosaReference::ConvertDType(dtype, true);

    // Tensor maps
    tosa_tensor_t* imp = nullptr;
    tosa_tensor_t* ref = nullptr;
    tosa_tensor_t* bnd = nullptr;

    // Verifier exit status
    tvf_status_t verifyStatus;

    verifyStatus = validateResultFile(impResultFile, impTosaRefType, npyShape);
    if (verifyStatus == tvf_status_t::TVF_COMPLIANT)
    {
        // Read implementation result numpy into a data buffer
        imp = new tosa_tensor_t;
        if (!createTensorMap(impResultFile, impTosaRefType, ofmName, npyShape, imp))
        {
            verifyStatus = tvf_status_t::TVF_ERROR;
        }
    }

    if (verifyStatus == tvf_status_t::TVF_COMPLIANT)
        verifyStatus = validateResultFile(refResultFile, refTosaRefType, npyShape);

    if (verifyStatus == tvf_status_t::TVF_COMPLIANT)
    {
        // Read reference result numpy into a data buffer
        ref = new tosa_tensor_t;
        if (!createTensorMap(refResultFile, refTosaRefType, ofmName, npyShape, ref))
        {
            verifyStatus = tvf_status_t::TVF_ERROR;
        }
    }

    if (!bndResultFile.empty())
    {
        if (verifyStatus == tvf_status_t::TVF_COMPLIANT)
            verifyStatus = validateResultFile(bndResultFile, refTosaRefType, npyShape);

        if (verifyStatus == tvf_status_t::TVF_COMPLIANT)
        {
            // Read reference bounds result numpy into a data buffer
            bnd = new tosa_tensor_t;
            if (!createTensorMap(bndResultFile, refTosaRefType, ofmName, npyShape, bnd))
            {
                verifyStatus = tvf_status_t::TVF_ERROR;
            }
        }
    }

    if (verifyStatus == tvf_status_t::TVF_COMPLIANT)
    {
        // Serialize the JSON object to a string
        std::string testDescString = test_desc["meta"]["compliance"].dump();

        // Check for compliant results
        if (!tvf_verify_data(ref, bnd, imp, testDescString.c_str()))
        {
            verifyStatus = tvf_status_t::TVF_NON_COMPLIANT;
        }
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
    if (verifyStatus == tvf_status_t::TVF_NON_COMPLIANT)
    {
        std::cout << "Test results: Failure - non-compliant" << std::endl;
    }
    else if (verifyStatus == tvf_status_t::TVF_COMPLIANT)
    {
        std::cout << "Test results: Pass - compliant" << std::endl;
    }
    return verifyStatus;
}

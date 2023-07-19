
// Copyright (c) 2020-2023, ARM Limited.
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

#include "tensor.h"
#include "arith_util.h"
#include "array_proxy.h"
#include "half.hpp"

using namespace TosaReference;
using namespace Eigen;
using namespace tosa;

TosaReference::Tensor::Tensor(const std::string tensorName_,
                              const DType serializationDtype_,
                              const std::vector<int> shape_)
    : tensorName(tensorName_)
    , serializationDtype(serializationDtype_)
    , shape(shape_)
    , tensorDtype(ConvertDType(serializationDtype_))
{
    producer = nullptr;
    isValid  = false;
    consumers.clear();
    isSubgraphInput     = false;
    isSubgraphOutput    = false;
    isParentGraphOutput = false;
}

TosaReference::Tensor::~Tensor()
{}

int TosaReference::Tensor::setIsParentGraphOutput()
{
    isParentGraphOutput = true;
    return 0;
}

int TosaReference::Tensor::setIsSubgraphInput()
{
    isSubgraphInput = true;
    return 0;
}

int TosaReference::Tensor::setIsSubgraphOutput()
{
    isSubgraphOutput = true;
    return 0;
}

int TosaReference::Tensor::setProducer(GraphNode* node)
{
    ASSERT_MSG(node, "Tensor::setProducer: no node passed in");
    ASSERT_MSG(!producer, "Tensor::setProducer: producer node already set, tensor %s", tensorName.c_str());
    producer = node;

    return 0;
}

int TosaReference::Tensor::addConsumer(GraphNode* node)
{
    ASSERT_MSG(node, "Tensor::addConsumer: no node passed in");
    consumers.push_back(node);

    return 0;
}

int TosaReference::Tensor::dumpTensorParams(FILE* out) const
{
    fprintf(out, "Name: %s DType=%s isValid=%d Rank=%d Shape=%s\n", tensorName.c_str(), EnumNameTOSAREFTYPE(getDtype()),
            getIsValid(), getRank(), getShapeAsString().c_str());

    return 0;
}

int TosaReference::Tensor::dumpTensorParams(std::ostream& out) const
{
    out << "Name: " << getName() << " DType=" << EnumNameTOSAREFTYPE(getDtype()) << " isValid=" << getIsValid()
        << " Rank=" << getRank() << " Shape=" << getShapeAsString() << "\n";

    return 0;
}

int TosaReference::Tensor::readFromNpyFile(const char* filename)
{
    uint32_t elements            = getElementCount();
    double* f64databuf           = nullptr;
    float* f32databuf            = nullptr;
    half_float::half* f16databuf = nullptr;
    int32_t* i32databuf          = nullptr;
    int64_t* i64databuf          = nullptr;
    bool* bdatabuf               = nullptr;
    NumpyUtilities::NPError nperror;
    TOSA_REF_TYPE dtype       = getDtype();
    DType serialization_dtype = getSerializationDtype();

    assert(dtype == ConvertDType(serialization_dtype));
    // if dtype is FP64, serialization_dtype must be one of FP32, FP16, BF16
    assert(dtype != TOSA_REF_TYPE_FP64 || serialization_dtype == DType_FP32 || serialization_dtype == DType_FP16 ||
           serialization_dtype == DType_BF16);

    switch (serialization_dtype)
    {
        case DType_FP32:
        case DType_BF16:
            f32databuf = (float*)calloc(sizeof(float), elements);
            ASSERT_MEM(f32databuf);

            nperror = NumpyUtilities::readFromNpyFile(filename, elements, f32databuf);
            break;
        case DType_FP16:
            f16databuf = (half_float::half*)calloc(sizeof(half_float::half), elements);
            ASSERT_MEM(f16databuf);

            nperror = NumpyUtilities::readFromNpyFile(filename, elements, f16databuf);
            break;
        case DType_INT32:
        case DType_UINT8:
        case DType_INT4:
        case DType_INT8:
        case DType_INT16:
        case DType_UINT16:
            i32databuf = (int32_t*)calloc(sizeof(int32_t), elements);
            ASSERT_MEM(i32databuf);

            nperror = NumpyUtilities::readFromNpyFile(filename, elements, i32databuf);
            break;
        case DType_INT48:
            i64databuf = (int64_t*)calloc(sizeof(int64_t), elements);
            ASSERT_MEM(i64databuf);

            nperror = NumpyUtilities::readFromNpyFile(filename, elements, i64databuf);
            break;
        case DType_BOOL:
            bdatabuf = (bool*)calloc(sizeof(bool), elements);
            ASSERT_MEM(bdatabuf);

            nperror = NumpyUtilities::readFromNpyFile(filename, elements, bdatabuf);
            break;
        default:
            FATAL_ERROR("unknown tensor type=%s", EnumNameDType(serialization_dtype));
    }

    switch (nperror)
    {
        case NumpyUtilities::NO_ERROR:
            break;
        case NumpyUtilities::FILE_NOT_FOUND:
            FATAL_ERROR("readFromNpyFile: Cannot open file %s", filename);
        case NumpyUtilities::FILE_IO_ERROR:
            FATAL_ERROR("readFromNpyFile: IO error reading file: %s", filename);
        case NumpyUtilities::FILE_TYPE_MISMATCH:
            FATAL_ERROR("readFromNpyFile: Tensor type %s and Numpy file type mismatch for tensor %s filename %s",
                        EnumNameTOSAREFTYPE(getDtype()), getName().c_str(), filename);
        case NumpyUtilities::HEADER_PARSE_ERROR:
            FATAL_ERROR("Numpy header parsing error for file: %s", filename);
        case NumpyUtilities::BUFFER_SIZE_MISMATCH:
            FATAL_ERROR("Buffer size does not match numpy file size for tensor %s filename %s", getName().c_str(),
                        filename);
        default:
            FATAL_ERROR("Unknown error parsing Numpy file: %s", filename);
    }

    switch (dtype)
    {
        case TOSA_REF_TYPE_FP16:
            // Convert from fp16 to fp32 so that fp16 values can be manipulated as float
            f32databuf = (float*)calloc(sizeof(float), elements);
            ASSERT_MEM(f32databuf);
            for (uint32_t i = 0; i < elements; i++)
            {
                f32databuf[i] = half_float::half_cast<float, half_float::half>(f16databuf[i]);
            }
            if (setTensorValueFloat(elements, f32databuf))
            {
                free(f16databuf);
                free(f32databuf);
                return 1;
            }
            break;
        case TOSA_REF_TYPE_BF16:
            for (uint32_t i = 0; i < elements; i++)
            {
                ASSERT_MSG(checkValidBFloat(f32databuf[i]), "Input float value not a valid bfloat16 value.");
            }
            if (setTensorValueFloat(elements, f32databuf))
            {
                free(f32databuf);
                return 1;
            }
            break;
        case TOSA_REF_TYPE_FP32:
            if (setTensorValueFloat(elements, f32databuf))
            {
                free(f32databuf);
                return 1;
            }
            break;
        case TOSA_REF_TYPE_INT32:
        case TOSA_REF_TYPE_UINT8:
        case TOSA_REF_TYPE_INT4:
        case TOSA_REF_TYPE_INT8:
        case TOSA_REF_TYPE_INT16:
        case TOSA_REF_TYPE_UINT16:
            if (setTensorValueInt32(elements, i32databuf))
            {
                free(i32databuf);
                return 1;
            }
            break;
        case TOSA_REF_TYPE_INT48:
            if (setTensorValueInt64(elements, i64databuf))
            {
                free(i64databuf);
                return 1;
            }
            break;
        case TOSA_REF_TYPE_BOOL:
            if (setTensorValueBool(elements, bdatabuf))
            {
                free(i32databuf);
                return 1;
            }
            break;
        case TOSA_REF_TYPE_FP64:
            switch (serialization_dtype)
            {
                case DType_FP16:
                    // FP16 -> FP64
                    f64databuf = (double*)calloc(sizeof(double), elements);
                    ASSERT_MEM(f64databuf);
                    for (uint32_t i = 0; i < elements; i++)
                    {
                        f64databuf[i] = half_float::half_cast<double, half_float::half>(f16databuf[i]);
                    }
                    if (setTensorValueDouble(elements, f64databuf))
                    {
                        free(f16databuf);
                        free(f64databuf);
                        return 1;
                    }
                    break;
                case DType_BF16:
                    // BF16 -> FP64
                    f64databuf = (double*)calloc(sizeof(double), elements);
                    ASSERT_MEM(f64databuf);
                    for (uint32_t i = 0; i < elements; i++)
                    {
                        ASSERT_MSG(checkValidBFloat(f32databuf[i]), "Input float value not a valid bfloat16 value.");
                        f64databuf[i] = static_cast<double>(f32databuf[i]);
                    }
                    if (setTensorValueDouble(elements, f64databuf))
                    {
                        free(f32databuf);
                        free(f64databuf);
                        return 1;
                    }
                    break;
                case DType_FP32:
                    // FP32 -> FP64
                    f64databuf = (double*)calloc(sizeof(double), elements);
                    ASSERT_MEM(f64databuf);
                    for (uint32_t i = 0; i < elements; i++)
                    {
                        f64databuf[i] = static_cast<double>(f32databuf[i]);
                    }
                    if (setTensorValueDouble(elements, f64databuf))
                    {
                        free(f32databuf);
                        free(f64databuf);
                        return 1;
                    }
                    break;
                default:
                    FATAL_ERROR("unexpected tensor type=%s and original tensor type=%s", EnumNameTOSAREFTYPE(dtype),
                                EnumNameDType(serialization_dtype));
            }
            break;
        default:
            FATAL_ERROR("unsupported tensor type=%s", EnumNameTOSAREFTYPE(dtype));
    }

    setIsValid();

    if (f32databuf)
        free(f32databuf);
    if (f16databuf)
        free(f16databuf);
    if (f64databuf)
        free(f64databuf);
    if (i32databuf)
        free(i32databuf);
    if (i64databuf)
        free(i64databuf);
    if (bdatabuf)
        free(bdatabuf);

    return 0;
}

int TosaReference::Tensor::writeToNpyFile(const char* filename) const
{
    float* f32databuf               = nullptr;
    double* f64databuf              = nullptr;
    half_float::half* f16databuf    = nullptr;
    int32_t* i32databuf             = nullptr;
    int64_t* i64databuf             = nullptr;
    bool* bdatabuf                  = nullptr;
    NumpyUtilities::NPError nperror = NumpyUtilities::NO_ERROR;
    uint32_t elements               = getElementCount();
    const TOSA_REF_TYPE dtype       = getDtype();

    switch (dtype)
    {
        case TOSA_REF_TYPE_FP32:
        case TOSA_REF_TYPE_BF16:
            f32databuf = (float*)calloc(sizeof(float), elements);
            ASSERT_MEM(f32databuf);

            if (getTensorValueFloat(elements, f32databuf))
            {
                free(f32databuf);
                return 1;
            }
            nperror = NumpyUtilities::writeToNpyFile(filename, shape, f32databuf);

            free(f32databuf);
            break;
        case TOSA_REF_TYPE_FP16:
            f32databuf = (float*)calloc(sizeof(float), elements);
            ASSERT_MEM(f32databuf);
            f16databuf = (half_float::half*)calloc(sizeof(half_float::half), elements);
            ASSERT_MEM(f16databuf);

            if (getTensorValueFloat(elements, f32databuf))
            {
                free(f32databuf);
                free(f16databuf);
                return 1;
            }
            // Convert fp32 to fp16 so that output file contains valid fp16 data
            for (uint32_t i = 0; i < elements; i++)
            {
                f16databuf[i] = half_float::half_cast<half_float::half, float>(f32databuf[i]);
            }
            nperror = NumpyUtilities::writeToNpyFile(filename, shape, f16databuf);

            free(f32databuf);
            free(f16databuf);
            break;
        case TOSA_REF_TYPE_INT32:
        case TOSA_REF_TYPE_UINT8:
        case TOSA_REF_TYPE_INT4:
        case TOSA_REF_TYPE_INT8:
        case TOSA_REF_TYPE_INT16:
        case TOSA_REF_TYPE_UINT16:
            i32databuf = (int32_t*)calloc(sizeof(int32_t), elements);
            ASSERT_MEM(i32databuf);

            if (getTensorValueInt32(elements, i32databuf))
            {
                free(i32databuf);
                return 1;
            }

            nperror = NumpyUtilities::writeToNpyFile(filename, shape, i32databuf);

            free(i32databuf);
            break;
        case TOSA_REF_TYPE_INT48:
            i64databuf = (int64_t*)calloc(sizeof(int64_t), elements);
            ASSERT_MEM(i64databuf);

            if (getTensorValueInt64(elements, i64databuf))
            {
                free(i64databuf);
                return 1;
            }

            nperror = NumpyUtilities::writeToNpyFile(filename, shape, i64databuf);

            free(i64databuf);
            break;
        case TOSA_REF_TYPE_BOOL:
            bdatabuf = (bool*)calloc(sizeof(bool), elements);
            ASSERT_MEM(bdatabuf);

            if (getTensorValueBool(elements, bdatabuf))
            {
                free(bdatabuf);
                return 1;
            }

            nperror = NumpyUtilities::writeToNpyFile(filename, shape, bdatabuf);

            free(bdatabuf);
            break;
        case TOSA_REF_TYPE_FP64:
            // @todo : support FP64 dtype
            f64databuf = (double*)calloc(sizeof(double), elements);
            ASSERT_MEM(f64databuf);

            if (getTensorValueDouble(elements, f64databuf))
            {
                free(f64databuf);
                return 1;
            }
            nperror = NumpyUtilities::writeToNpyFile(filename, shape, f64databuf);

            free(f64databuf);
            break;
        case TOSA_REF_TYPE_UNKNOWN:
            FATAL_ERROR("unsupported tensor type=%s", EnumNameTOSAREFTYPE(getDtype()));
    }

    switch (nperror)
    {
        case NumpyUtilities::NO_ERROR:
            break;
        case NumpyUtilities::FILE_NOT_FOUND:
            FATAL_ERROR("writeToNpyFile: Cannot open output file %s", filename);
        case NumpyUtilities::FILE_IO_ERROR:
            FATAL_ERROR("writeToNpyFile: IO error writing file: %s", filename);
        case NumpyUtilities::FILE_TYPE_MISMATCH:
            FATAL_ERROR("writeToNpyFile: Tensor type and Numpy file type mismatch for tensor %s filename %s",
                        getName().c_str(), filename);
        case NumpyUtilities::HEADER_PARSE_ERROR:
            FATAL_ERROR("Numpy header parsing error for file: %s", filename);
        case NumpyUtilities::BUFFER_SIZE_MISMATCH:
            FATAL_ERROR("Buffer size does not match numpy file size for tensor %s filename %s", getName().c_str(),
                        filename);
        default:
            FATAL_ERROR("Unknown error writing Numpy file: %s", filename);
    }

    return 0;
}

template <class T>
int TosaReference::TensorTemplate<T>::copyValueFrom(TosaReference::Tensor* src)
{
    FATAL_ERROR("TensorTemplate<T>::copyValueFrom should not be called.  "
                "Implement template specialization version.");
    return 0;
}

#define DEF_CTENSOR_COPY_VALUE_FROM(RANK, TYPE)                                                                        \
    template <>                                                                                                        \
    int TosaReference::Tensor##RANK<TYPE>::copyValueFrom(TosaReference::Tensor* src)                                   \
    {                                                                                                                  \
        TosaReference::Tensor##RANK<TYPE>* t = dynamic_cast<Tensor##RANK<TYPE>*>(src);                                 \
        if (!t)                                                                                                        \
        {                                                                                                              \
            WARNING("tensor %s templated class does not match %s", src->getName().c_str(), this->getName().c_str());   \
            return 1;                                                                                                  \
        }                                                                                                              \
                                                                                                                       \
        const uint32_t src_rank       = src->getRank();                                                                \
        const uint32_t dst_rank       = this->getRank();                                                               \
        const TOSA_REF_TYPE src_dtype = src->getDtype();                                                               \
        const TOSA_REF_TYPE dst_dtype = this->getDtype();                                                              \
        bool tensor_match             = true;                                                                          \
                                                                                                                       \
        if ((src_rank != dst_rank) || (src_dtype != dst_dtype))                                                        \
        {                                                                                                              \
            tensor_match = false;                                                                                      \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            for (uint32_t i = 0; i < src_rank; i++)                                                                    \
            {                                                                                                          \
                int src_dim = src->getShape()[i];                                                                      \
                int dst_dim = this->getShape()[i];                                                                     \
                if (src_dim != dst_dim)                                                                                \
                {                                                                                                      \
                    tensor_match = false;                                                                              \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
                                                                                                                       \
        if (!tensor_match)                                                                                             \
        {                                                                                                              \
            WARNING("source tensor %s (rank=%u, dtype=%s, shape=%s) doesn't match destination tensor %s (rank=%u, "    \
                    "dtype=%s, shape=%s)",                                                                             \
                    src->getName().c_str(), src_rank, EnumNameTOSAREFTYPE(src_dtype), src->getShapeAsString().c_str(), \
                    this->getName().c_str(), dst_rank, EnumNameTOSAREFTYPE(dst_dtype),                                 \
                    this->getShapeAsString().c_str());                                                                 \
            return 1;                                                                                                  \
        }                                                                                                              \
                                                                                                                       \
        this->getTensor() = t->getTensor();                                                                            \
        return 0;                                                                                                      \
    }

DEF_CTENSOR_COPY_VALUE_FROM(0, float)
DEF_CTENSOR_COPY_VALUE_FROM(1, float)
DEF_CTENSOR_COPY_VALUE_FROM(2, float)
DEF_CTENSOR_COPY_VALUE_FROM(3, float)
DEF_CTENSOR_COPY_VALUE_FROM(4, float)
DEF_CTENSOR_COPY_VALUE_FROM(5, float)
DEF_CTENSOR_COPY_VALUE_FROM(6, float)
DEF_CTENSOR_COPY_VALUE_FROM(0, double)
DEF_CTENSOR_COPY_VALUE_FROM(1, double)
DEF_CTENSOR_COPY_VALUE_FROM(2, double)
DEF_CTENSOR_COPY_VALUE_FROM(3, double)
DEF_CTENSOR_COPY_VALUE_FROM(4, double)
DEF_CTENSOR_COPY_VALUE_FROM(5, double)
DEF_CTENSOR_COPY_VALUE_FROM(6, double)
DEF_CTENSOR_COPY_VALUE_FROM(0, int32_t)
DEF_CTENSOR_COPY_VALUE_FROM(1, int32_t)
DEF_CTENSOR_COPY_VALUE_FROM(2, int32_t)
DEF_CTENSOR_COPY_VALUE_FROM(3, int32_t)
DEF_CTENSOR_COPY_VALUE_FROM(4, int32_t)
DEF_CTENSOR_COPY_VALUE_FROM(5, int32_t)
DEF_CTENSOR_COPY_VALUE_FROM(6, int32_t)
DEF_CTENSOR_COPY_VALUE_FROM(0, int64_t)
DEF_CTENSOR_COPY_VALUE_FROM(1, int64_t)
DEF_CTENSOR_COPY_VALUE_FROM(2, int64_t)
DEF_CTENSOR_COPY_VALUE_FROM(3, int64_t)
DEF_CTENSOR_COPY_VALUE_FROM(4, int64_t)
DEF_CTENSOR_COPY_VALUE_FROM(5, int64_t)
DEF_CTENSOR_COPY_VALUE_FROM(6, int64_t)
DEF_CTENSOR_COPY_VALUE_FROM(0, bool)
DEF_CTENSOR_COPY_VALUE_FROM(1, bool)
DEF_CTENSOR_COPY_VALUE_FROM(2, bool)
DEF_CTENSOR_COPY_VALUE_FROM(3, bool)
DEF_CTENSOR_COPY_VALUE_FROM(4, bool)
DEF_CTENSOR_COPY_VALUE_FROM(5, bool)
DEF_CTENSOR_COPY_VALUE_FROM(6, bool)

#undef DEF_CTENSOR_COPY_VALUE_FROM

int TosaReference::Tensor::readfromVector(const ArrayProxy<double> vals)
{
    uint32_t elements = getElementCount();
    switch (getDtype())
    {
        case TOSA_REF_TYPE_FP64:
            if (vals.size() != elements)
            {
                WARNING("The input size (%ld) doesn't match the number of elements (%d) assigned to the tensor.",
                        vals.size(), elements);
                return -1;
            }

            setTensorValueDouble(elements, vals.data());
            break;
        default:
            WARNING("The input type (float) doesn't match the data type assigned to the tensor (%s).",
                    EnumNameTOSAREFTYPE(getDtype()));
            return -2;
    }
    setIsValid();
    return 0;
}

int TosaReference::Tensor::readfromVector(const ArrayProxy<float> vals)
{
    uint32_t elements = getElementCount();
    switch (getDtype())
    {
        case TOSA_REF_TYPE_FP16:
        case TOSA_REF_TYPE_FP32:
            if (vals.size() != elements)
            {
                WARNING("The input size (%ld) doesn't match the number of elements (%d) assigned to the tensor.",
                        vals.size(), elements);
                return -1;
            }

            setTensorValueFloat(elements, vals.data());
            break;
        case TOSA_REF_TYPE_BF16:
            if (vals.size() != elements)
            {
                WARNING("The input size (%ld) doesn't match the number of elements (%d) assigned to the tensor.",
                        vals.size(), elements);
                return -1;
            }

            for (auto v : vals)
            {
                ASSERT_MSG(checkValidBFloat(v), "Input float value not a valid bfloat16 value.");
            }

            setTensorValueFloat(elements, vals.data());
            break;
        default:
            WARNING("The input type (float) doesn't match the data type assigned to the tensor (%s).",
                    EnumNameTOSAREFTYPE(getDtype()));
            return -2;
    }
    setIsValid();
    return 0;
}

int TosaReference::Tensor::readfromVector(const ArrayProxy<half_float::half> vals)
{
    uint32_t elements = getElementCount();
    std::vector<float> tensor(elements);

    switch (getDtype())
    {
        case TOSA_REF_TYPE_FP16:
            if (vals.size() != elements)
            {
                WARNING("The input size (%ld) doesn't match the number of elements (%d) assigned to the tensor.",
                        vals.size(), elements);
                return -1;
            }

            // Convert from fp16 to fp32
            for (uint32_t i = 0; i < elements; i++)
            {
                tensor[i] = half_float::half_cast<float, half_float::half>(vals[i]);
            }

            setTensorValueFloat(elements, tensor.data());
            break;
        default:
            WARNING("The input type doesn't match the data type assigned to the tensor (%s).",
                    EnumNameTOSAREFTYPE(getDtype()));
            return -2;
    }
    setIsValid();
    return 0;
}

int TosaReference::Tensor::readfromVector(const ArrayProxy<int32_t> vals)
{
    uint32_t elements = getElementCount();
    switch (getDtype())
    {
        case TOSA_REF_TYPE_INT32:
        case TOSA_REF_TYPE_UINT8:
        case TOSA_REF_TYPE_INT4:
        case TOSA_REF_TYPE_INT8:
        case TOSA_REF_TYPE_INT16:
        case TOSA_REF_TYPE_UINT16:
            if (vals.size() != elements)
            {
                WARNING("The input size (%ld) doesn't match the number of elements (%d) assigned to the tensor.",
                        vals.size(), elements);
                return -1;
            }

            setTensorValueInt32(elements, vals.data());
            break;
        default:
            WARNING("The input type doesn't match the data type assigned to the tensor (%s).",
                    EnumNameTOSAREFTYPE(getDtype()));
            return -2;
    }
    setIsValid();
    return 0;
}

int TosaReference::Tensor::readfromVector(const ArrayProxy<int64_t> vals)
{
    uint32_t elements = getElementCount();
    switch (getDtype())
    {
        case TOSA_REF_TYPE_INT48:
            if (vals.size() != elements)
            {
                WARNING("The input size (%ld) doesn't match the number of elements (%d) assigned to the tensor.",
                        vals.size(), elements);
                return -1;
            }

            setTensorValueInt64(elements, vals.data());
            break;
        default:
            WARNING("The input type doesn't match the data type assigned to the tensor (%s).",
                    EnumNameTOSAREFTYPE(getDtype()));
            return -2;
    }
    setIsValid();
    return 0;
}

int TosaReference::Tensor::readfromVector(const ArrayProxy<unsigned char> vals)
{
    uint32_t elements = getElementCount();

    switch (getDtype())
    {
        case TOSA_REF_TYPE_BOOL:
            if (vals.size() != elements)
            {
                WARNING("The input size (%ld) doesn't match the number of elements (%d) assigned to the tensor.",
                        vals.size(), elements);
                return -1;
            }

            setTensorValueBool(elements, reinterpret_cast<const bool*>(vals.data()));
            break;
        default:
            WARNING("The input type (bool) doesn't match the data type assigned to the tensor (%s).",
                    EnumNameTOSAREFTYPE(getDtype()));
            return -2;
    }
    setIsValid();
    return 0;
}

int TosaReference::Tensor::writeToVector(ArrayProxy<double> vals)
{
    uint32_t elements = getElementCount();

    switch (getDtype())
    {
        case TOSA_REF_TYPE_FP64:
            if (vals.size() != elements)
            {
                WARNING("The output size (%ld) doesn't match the number of elements (%d) assigned to the tensor.",
                        vals.size(), elements);
                return -1;
            }

            getTensorValueDouble(elements, vals.data());
            break;
        default:
            WARNING("The output type (float) doesn't match the data type assigned to the tensor (%s).",
                    EnumNameTOSAREFTYPE(getDtype()));
            return -2;
    }
    return 0;
}

int TosaReference::Tensor::writeToVector(ArrayProxy<float> vals)
{
    uint32_t elements = getElementCount();

    switch (getDtype())
    {
        case TOSA_REF_TYPE_FP16:
        case TOSA_REF_TYPE_FP32:
            if (vals.size() != elements)
            {
                WARNING("The output size (%ld) doesn't match the number of elements (%d) assigned to the tensor.",
                        vals.size(), elements);
                return -1;
            }

            getTensorValueFloat(elements, vals.data());
            break;
        case TOSA_REF_TYPE_BF16:
            if (vals.size() != elements)
            {
                WARNING("The output size (%ld) doesn't match the number of elements (%d) assigned to the tensor.",
                        vals.size(), elements);
                return -1;
            }

            getTensorValueFloat(elements, vals.data());

            for (auto v : vals)
            {
                ASSERT_MSG(checkValidBFloat(v), "Float value not a valid bfloat16 value.");
            }

            break;
        default:
            WARNING("The output type (float) doesn't match the data type assigned to the tensor (%s).",
                    EnumNameTOSAREFTYPE(getDtype()));
            return -2;
    }
    return 0;
}

int TosaReference::Tensor::writeToVector(ArrayProxy<half_float::half> vals)
{
    uint32_t elements = getElementCount();
    std::vector<float> tensor(elements);

    switch (getDtype())
    {
        case TOSA_REF_TYPE_FP16:
            if (vals.size() != elements)
            {
                WARNING("The output size (%ld) doesn't match the number of elements (%d) assigned to the tensor.",
                        vals.size(), elements);
                return -1;
            }

            getTensorValueFloat(elements, tensor.data());

            // Convert fp32 to fp16
            for (uint32_t i = 0; i < elements; i++)
            {
                vals[i] = half_float::half_cast<half_float::half, float>(tensor[i]);
            }
            break;
        default:
            WARNING("The output type doesn't match the data type assigned to the tensor (%s).",
                    EnumNameTOSAREFTYPE(getDtype()));
            return -2;
    }
    return 0;
}

int TosaReference::Tensor::writeToVector(ArrayProxy<int32_t> vals)
{
    uint32_t elements = getElementCount();

    switch (getDtype())
    {
        case TOSA_REF_TYPE_INT32:
        case TOSA_REF_TYPE_UINT8:
        case TOSA_REF_TYPE_INT4:
        case TOSA_REF_TYPE_INT8:
        case TOSA_REF_TYPE_INT16:
        case TOSA_REF_TYPE_UINT16:
            if (vals.size() != elements)
            {
                WARNING("The output size (%ld) doesn't match the number of elements (%d) assigned to the tensor.",
                        vals.size(), elements);
                return -1;
            }

            getTensorValueInt32(elements, vals.data());
            break;
        default:
            WARNING("The output type doesn't match the data type assigned to the tensor (%s).",
                    EnumNameTOSAREFTYPE(getDtype()));
            return -2;
    }
    return 0;
}

int TosaReference::Tensor::writeToVector(ArrayProxy<int64_t> vals)
{
    uint32_t elements = getElementCount();

    switch (getDtype())
    {
        case TOSA_REF_TYPE_INT48:
            if (vals.size() != elements)
            {
                WARNING("The output size (%ld) doesn't match the number of elements (%d) assigned to the tensor.",
                        vals.size(), elements);
                return -1;
            }

            getTensorValueInt64(elements, vals.data());
            break;
        default:
            WARNING("The output type doesn't match the data type assigned to the tensor (%s).",
                    EnumNameTOSAREFTYPE(getDtype()));
            return -2;
    }
    return 0;
}

int TosaReference::Tensor::writeToVector(ArrayProxy<unsigned char> vals)
{
    uint32_t elements = getElementCount();

    switch (getDtype())
    {
        case TOSA_REF_TYPE_BOOL:
            if (vals.size() != elements)
            {
                WARNING("The output size (%ld) doesn't match the number of elements (%d) assigned to the tensor.",
                        vals.size(), elements);
                return -1;
            }

            getTensorValueBool(elements, reinterpret_cast<bool*>(vals.data()));
            break;
        default:
            WARNING("The output type (bool) doesn't match the data type assigned to the tensor (%s).",
                    EnumNameTOSAREFTYPE(getDtype()));
            return -2;
    }
    return 0;
}

template <class T>
int TosaReference::TensorTemplate<T>::setTensorValueDouble(const size_t buflen, const double* vals)
{
    FATAL_ERROR("TensorTemplate<T>::setTensorValueFloat should not be called.  "
                "Implement template specialization version.");
    return 0;
}

template <>
int TosaReference::Tensor0<double>::setTensorValueDouble(const size_t bufLen, const double* vals)
{
    ASSERT_MSG(bufLen == getElementCount(), "Total elements must match");

    (*tensor)(0) = vals[0];

    return 0;
}

template <>
int TosaReference::Tensor1<double>::setTensorValueDouble(const size_t bufLen, const double* vals)
{
    uint32_t idx = 0;

    ASSERT_MSG(bufLen == getElementCount(), "Total elements must match");

    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        (*tensor)(i0) = vals[idx++];
    }

    return 0;
}

template <>
int TosaReference::Tensor2<double>::setTensorValueDouble(const size_t bufLen, const double* vals)
{
    uint32_t idx = 0;

    ASSERT_MSG(bufLen == getElementCount(), "Total elements must match");

    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            (*tensor)(i0, i1) = vals[idx++];
        }
    }

    return 0;
}

template <>
int TosaReference::Tensor3<double>::setTensorValueDouble(const size_t bufLen, const double* vals)
{
    uint32_t idx = 0;

    ASSERT_MSG(bufLen == getElementCount(), "Total elements must match");

    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            for (int i2 = 0; i2 < shape[2]; i2++)
            {
                (*tensor)(i0, i1, i2) = vals[idx++];
            }
        }
    }

    return 0;
}

template <>
int TosaReference::Tensor4<double>::setTensorValueDouble(const size_t bufLen, const double* vals)
{
    uint32_t idx = 0;

    ASSERT_MSG(bufLen == getElementCount(), "Total elements must match");

    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            for (int i2 = 0; i2 < shape[2]; i2++)
            {
                for (int i3 = 0; i3 < shape[3]; i3++)
                {
                    (*tensor)(i0, i1, i2, i3) = vals[idx++];
                }
            }
        }
    }

    return 0;
}

template <>
int TosaReference::Tensor5<double>::setTensorValueDouble(const size_t bufLen, const double* vals)
{
    uint32_t idx = 0;

    ASSERT_MSG(bufLen == getElementCount(), "Total elements must match");

    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            for (int i2 = 0; i2 < shape[2]; i2++)
            {
                for (int i3 = 0; i3 < shape[3]; i3++)
                {
                    for (int i4 = 0; i4 < shape[4]; i4++)
                    {
                        (*tensor)(i0, i1, i2, i3, i4) = vals[idx++];
                    }
                }
            }
        }
    }

    return 0;
}

template <>
int TosaReference::Tensor6<double>::setTensorValueDouble(const size_t bufLen, const double* vals)
{
    uint32_t idx = 0;

    ASSERT_MSG(bufLen == getElementCount(), "Total elements must match");

    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            for (int i2 = 0; i2 < shape[2]; i2++)
            {
                for (int i3 = 0; i3 < shape[3]; i3++)
                {
                    for (int i4 = 0; i4 < shape[4]; i4++)
                    {
                        for (int i5 = 0; i5 < shape[5]; i5++)
                        {
                            (*tensor)(i0, i1, i2, i3, i4, i5) = vals[idx++];
                        }
                    }
                }
            }
        }
    }
    return 0;
}

template <class T>
int TosaReference::TensorTemplate<T>::setTensorValueFloat(const size_t buflen, const float* vals)
{
    FATAL_ERROR("TensorTemplate<T>::setTensorValueFloat should not be called.  "
                "Implement template specialization version.");
    return 0;
}

template <>
int TosaReference::Tensor0<float>::setTensorValueFloat(const size_t bufLen, const float* vals)
{
    ASSERT_MSG(bufLen == getElementCount(), "Total elements must match");

    (*tensor)(0) = vals[0];

    return 0;
}

template <>
int TosaReference::Tensor1<float>::setTensorValueFloat(const size_t bufLen, const float* vals)
{
    uint32_t idx = 0;

    ASSERT_MSG(bufLen == getElementCount(), "Total elements must match");

    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        (*tensor)(i0) = vals[idx++];
    }

    return 0;
}

template <>
int TosaReference::Tensor2<float>::setTensorValueFloat(const size_t bufLen, const float* vals)
{
    uint32_t idx = 0;

    ASSERT_MSG(bufLen == getElementCount(), "Total elements must match");

    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            (*tensor)(i0, i1) = vals[idx++];
        }
    }

    return 0;
}

template <>
int TosaReference::Tensor3<float>::setTensorValueFloat(const size_t bufLen, const float* vals)
{
    uint32_t idx = 0;

    ASSERT_MSG(bufLen == getElementCount(), "Total elements must match");

    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            for (int i2 = 0; i2 < shape[2]; i2++)
            {
                (*tensor)(i0, i1, i2) = vals[idx++];
            }
        }
    }

    return 0;
}

template <>
int TosaReference::Tensor4<float>::setTensorValueFloat(const size_t bufLen, const float* vals)
{
    uint32_t idx = 0;

    ASSERT_MSG(bufLen == getElementCount(), "Total elements must match");

    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            for (int i2 = 0; i2 < shape[2]; i2++)
            {
                for (int i3 = 0; i3 < shape[3]; i3++)
                {
                    (*tensor)(i0, i1, i2, i3) = vals[idx++];
                }
            }
        }
    }

    return 0;
}

template <>
int TosaReference::Tensor5<float>::setTensorValueFloat(const size_t bufLen, const float* vals)
{
    uint32_t idx = 0;

    ASSERT_MSG(bufLen == getElementCount(), "Total elements must match");

    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            for (int i2 = 0; i2 < shape[2]; i2++)
            {
                for (int i3 = 0; i3 < shape[3]; i3++)
                {
                    for (int i4 = 0; i4 < shape[4]; i4++)
                    {
                        (*tensor)(i0, i1, i2, i3, i4) = vals[idx++];
                    }
                }
            }
        }
    }

    return 0;
}

template <>
int TosaReference::Tensor6<float>::setTensorValueFloat(const size_t bufLen, const float* vals)
{
    uint32_t idx = 0;

    ASSERT_MSG(bufLen == getElementCount(), "Total elements must match");

    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            for (int i2 = 0; i2 < shape[2]; i2++)
            {
                for (int i3 = 0; i3 < shape[3]; i3++)
                {
                    for (int i4 = 0; i4 < shape[4]; i4++)
                    {
                        for (int i5 = 0; i5 < shape[5]; i5++)
                        {
                            (*tensor)(i0, i1, i2, i3, i4, i5) = vals[idx++];
                        }
                    }
                }
            }
        }
    }
    return 0;
}

template <class T>
int TosaReference::TensorTemplate<T>::setTensorValueInt32(const size_t bufLen, const int32_t* vals)
{
    FATAL_ERROR("TensorTemplate<T>::setTensorValueInt32 should not be called.  "
                "Implement template specialization version.");
    return 0;
}

template <>
int TosaReference::Tensor0<int32_t>::setTensorValueInt32(const size_t bufLen, const int32_t* vals)
{
    ASSERT_MSG(bufLen == getElementCount(), "Total elements must match");

    (*tensor)(0) = vals[0];

    return 0;
}

template <>
int TosaReference::Tensor1<int32_t>::setTensorValueInt32(const size_t bufLen, const int32_t* vals)
{
    uint32_t idx = 0;

    ASSERT_MSG(bufLen == getElementCount(), "Total elements must match");

    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        (*tensor)(i0) = vals[idx++];
    }

    return 0;
}

template <>
int TosaReference::Tensor2<int32_t>::setTensorValueInt32(const size_t bufLen, const int32_t* vals)
{
    uint32_t idx = 0;

    ASSERT_MSG(bufLen == getElementCount(), "Total elements must match");

    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            (*tensor)(i0, i1) = vals[idx++];
        }
    }

    return 0;
}

template <>
int TosaReference::Tensor3<int32_t>::setTensorValueInt32(const size_t bufLen, const int32_t* vals)
{
    uint32_t idx = 0;

    ASSERT_MSG(bufLen == getElementCount(), "Total elements must match");

    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            for (int i2 = 0; i2 < shape[2]; i2++)
            {
                (*tensor)(i0, i1, i2) = vals[idx++];
            }
        }
    }

    return 0;
}

template <>
int TosaReference::Tensor4<int32_t>::setTensorValueInt32(const size_t bufLen, const int32_t* vals)
{
    uint32_t idx = 0;

    ASSERT_MSG(bufLen == getElementCount(), "Total elements must match");

    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            for (int i2 = 0; i2 < shape[2]; i2++)
            {
                for (int i3 = 0; i3 < shape[3]; i3++)
                {
                    (*tensor)(i0, i1, i2, i3) = vals[idx++];
                }
            }
        }
    }

    return 0;
}

template <>
int TosaReference::Tensor5<int32_t>::setTensorValueInt32(const size_t bufLen, const int32_t* vals)
{
    uint32_t idx = 0;

    ASSERT_MSG(bufLen == getElementCount(), "Total elements must match");

    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            for (int i2 = 0; i2 < shape[2]; i2++)
            {
                for (int i3 = 0; i3 < shape[3]; i3++)
                {
                    for (int i4 = 0; i4 < shape[4]; i4++)
                    {
                        (*tensor)(i0, i1, i2, i3, i4) = vals[idx++];
                    }
                }
            }
        }
    }

    return 0;
}

template <>
int TosaReference::Tensor6<int32_t>::setTensorValueInt32(const size_t bufLen, const int32_t* vals)
{
    uint32_t idx = 0;

    ASSERT_MSG(bufLen == getElementCount(), "Total elements must match");

    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            for (int i2 = 0; i2 < shape[2]; i2++)
            {
                for (int i3 = 0; i3 < shape[3]; i3++)
                {
                    for (int i4 = 0; i4 < shape[4]; i4++)
                    {
                        for (int i5 = 0; i5 < shape[5]; i5++)
                        {
                            (*tensor)(i0, i1, i2, i3, i4, i5) = vals[idx++];
                        }
                    }
                }
            }
        }
    }
    return 0;
}

template <class T>
int TosaReference::TensorTemplate<T>::setTensorValueInt64(const size_t bufLen, const int64_t* vals)
{
    FATAL_ERROR("TensorTemplate<T>::setTensorValueInt64 should not be called.  "
                "Implement template specialization version.");
    return 0;
}

template <>
int TosaReference::Tensor0<int64_t>::setTensorValueInt64(const size_t bufLen, const int64_t* vals)
{
    ASSERT_MSG(bufLen == getElementCount(), "Total elements must match");

    (*tensor)(0) = vals[0];

    return 0;
}

template <>
int TosaReference::Tensor1<int64_t>::setTensorValueInt64(const size_t bufLen, const int64_t* vals)
{
    uint32_t idx = 0;

    ASSERT_MSG(bufLen == getElementCount(), "Total elements must match");

    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        (*tensor)(i0) = vals[idx++];
    }

    return 0;
}

template <>
int TosaReference::Tensor2<int64_t>::setTensorValueInt64(const size_t bufLen, const int64_t* vals)
{
    uint32_t idx = 0;

    ASSERT_MSG(bufLen == getElementCount(), "Total elements must match");

    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            (*tensor)(i0, i1) = vals[idx++];
        }
    }

    return 0;
}

template <>
int TosaReference::Tensor3<int64_t>::setTensorValueInt64(const size_t bufLen, const int64_t* vals)
{
    uint32_t idx = 0;

    ASSERT_MSG(bufLen == getElementCount(), "Total elements must match");

    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            for (int i2 = 0; i2 < shape[2]; i2++)
            {
                (*tensor)(i0, i1, i2) = vals[idx++];
            }
        }
    }

    return 0;
}

template <>
int TosaReference::Tensor4<int64_t>::setTensorValueInt64(const size_t bufLen, const int64_t* vals)
{
    uint32_t idx = 0;

    ASSERT_MSG(bufLen == getElementCount(), "Total elements must match");

    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            for (int i2 = 0; i2 < shape[2]; i2++)
            {
                for (int i3 = 0; i3 < shape[3]; i3++)
                {
                    (*tensor)(i0, i1, i2, i3) = vals[idx++];
                }
            }
        }
    }

    return 0;
}

template <>
int TosaReference::Tensor5<int64_t>::setTensorValueInt64(const size_t bufLen, const int64_t* vals)
{
    uint32_t idx = 0;

    ASSERT_MSG(bufLen == getElementCount(), "Total elements must match");

    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            for (int i2 = 0; i2 < shape[2]; i2++)
            {
                for (int i3 = 0; i3 < shape[3]; i3++)
                {
                    for (int i4 = 0; i4 < shape[4]; i4++)
                    {
                        (*tensor)(i0, i1, i2, i3, i4) = vals[idx++];
                    }
                }
            }
        }
    }

    return 0;
}

template <>
int TosaReference::Tensor6<int64_t>::setTensorValueInt64(const size_t bufLen, const int64_t* vals)
{
    uint32_t idx = 0;

    ASSERT_MSG(bufLen == getElementCount(), "Total elements must match");

    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            for (int i2 = 0; i2 < shape[2]; i2++)
            {
                for (int i3 = 0; i3 < shape[3]; i3++)
                {
                    for (int i4 = 0; i4 < shape[4]; i4++)
                    {
                        for (int i5 = 0; i5 < shape[5]; i5++)
                        {
                            (*tensor)(i0, i1, i2, i3, i4, i5) = vals[idx++];
                        }
                    }
                }
            }
        }
    }
    return 0;
}

template <class T>
int TosaReference::TensorTemplate<T>::setTensorValueBool(const size_t buflen, const bool* vals)
{
    FATAL_ERROR("TensorTemplate<T>::setTensorValueBool should not be called.  "
                "Implement template specialization version.");
    return 0;
}

template <>
int TosaReference::Tensor0<bool>::setTensorValueBool(const size_t bufLen, const bool* vals)
{
    ASSERT_MSG(bufLen == getElementCount(), "Total elements must match");

    (*tensor)(0) = vals[0];

    return 0;
}

template <>
int TosaReference::Tensor1<bool>::setTensorValueBool(const size_t bufLen, const bool* vals)
{
    uint32_t idx = 0;

    ASSERT_MSG(bufLen == getElementCount(), "Total elements must match");

    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        (*tensor)(i0) = vals[idx++];
    }

    return 0;
}

template <>
int TosaReference::Tensor2<bool>::setTensorValueBool(const size_t bufLen, const bool* vals)
{
    uint32_t idx = 0;

    ASSERT_MSG(bufLen == getElementCount(), "Total elements must match");

    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            (*tensor)(i0, i1) = vals[idx++];
        }
    }

    return 0;
}

template <>
int TosaReference::Tensor3<bool>::setTensorValueBool(const size_t bufLen, const bool* vals)
{
    uint32_t idx = 0;

    ASSERT_MSG(bufLen == getElementCount(), "Total elements must match");

    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            for (int i2 = 0; i2 < shape[2]; i2++)
            {
                (*tensor)(i0, i1, i2) = vals[idx++];
            }
        }
    }

    return 0;
}

template <>
int TosaReference::Tensor4<bool>::setTensorValueBool(const size_t bufLen, const bool* vals)
{
    uint32_t idx = 0;

    ASSERT_MSG(bufLen == getElementCount(), "Total elements must match");

    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            for (int i2 = 0; i2 < shape[2]; i2++)
            {
                for (int i3 = 0; i3 < shape[3]; i3++)
                {
                    (*tensor)(i0, i1, i2, i3) = vals[idx++];
                }
            }
        }
    }

    return 0;
}

template <>
int TosaReference::Tensor5<bool>::setTensorValueBool(const size_t bufLen, const bool* vals)
{
    uint32_t idx = 0;

    ASSERT_MSG(bufLen == getElementCount(), "Total elements must match");

    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            for (int i2 = 0; i2 < shape[2]; i2++)
            {
                for (int i3 = 0; i3 < shape[3]; i3++)
                {
                    for (int i4 = 0; i4 < shape[4]; i4++)
                    {
                        (*tensor)(i0, i1, i2, i3, i4) = vals[idx++];
                    }
                }
            }
        }
    }

    return 0;
}

template <>
int TosaReference::Tensor6<bool>::setTensorValueBool(const size_t bufLen, const bool* vals)
{
    uint32_t idx = 0;

    ASSERT_MSG(bufLen == getElementCount(), "Total elements must match");

    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            for (int i2 = 0; i2 < shape[2]; i2++)
            {
                for (int i3 = 0; i3 < shape[3]; i3++)
                {
                    for (int i4 = 0; i4 < shape[4]; i4++)
                    {
                        for (int i5 = 0; i5 < shape[5]; i5++)
                        {
                            (*tensor)(i0, i1, i2, i3, i4, i5) = vals[idx++];
                        }
                    }
                }
            }
        }
    }
    return 0;
}

template <class T>
int TosaReference::TensorTemplate<T>::getTensorValueDouble(const size_t bufLen, double* vals) const
{
    FATAL_ERROR("TensorTemplate<T>::getTensorValueDouble should not be called.  "
                "Implement template specialization version.");
    return 0;
}

template <>
int TosaReference::Tensor0<double>::getTensorValueDouble(const size_t bufLen, double* vals) const
{
    int totalVals = 1;

    ASSERT_MSG((size_t)totalVals == bufLen, "Output buffer and tensor size do not match");

    vals[0] = (*tensor)(0);

    return 0;
}

template <>
int TosaReference::Tensor1<double>::getTensorValueDouble(const size_t bufLen, double* vals) const
{
    uint32_t idx  = 0;
    int totalVals = 1;

    for (size_t i = 0; i < shape.size(); i++)
    {
        totalVals *= shape[i];
    }

    ASSERT_MSG((size_t)totalVals == bufLen, "Output buffer and tensor size do not match");

    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        vals[idx++] = (*tensor)(i0);
    }

    return 0;
}

template <>
int TosaReference::Tensor2<double>::getTensorValueDouble(const size_t bufLen, double* vals) const
{
    uint32_t idx  = 0;
    int totalVals = 1;

    for (size_t i = 0; i < shape.size(); i++)
    {
        totalVals *= shape[i];
    }

    ASSERT_MSG((size_t)totalVals == bufLen, "Output buffer and tensor size do not match");

    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            vals[idx++] = (*tensor)(i0, i1);
        }
    }

    return 0;
}

template <>
int TosaReference::Tensor3<double>::getTensorValueDouble(const size_t bufLen, double* vals) const
{
    uint32_t idx  = 0;
    int totalVals = 1;

    for (size_t i = 0; i < shape.size(); i++)
    {
        totalVals *= shape[i];
    }

    ASSERT_MSG((size_t)totalVals == bufLen, "Output buffer and tensor size do not match");

    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            for (int i2 = 0; i2 < shape[2]; i2++)
            {
                vals[idx++] = (*tensor)(i0, i1, i2);
            }
        }
    }

    return 0;
}

template <>
int TosaReference::Tensor4<double>::getTensorValueDouble(const size_t bufLen, double* vals) const
{
    uint32_t idx  = 0;
    int totalVals = 1;

    for (size_t i = 0; i < shape.size(); i++)
    {
        totalVals *= shape[i];
    }

    ASSERT_MSG((size_t)totalVals == bufLen, "Output buffer and tensor size do not match");

    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            for (int i2 = 0; i2 < shape[2]; i2++)
            {
                for (int i3 = 0; i3 < shape[3]; i3++)
                {
                    vals[idx++] = (*tensor)(i0, i1, i2, i3);
                }
            }
        }
    }

    return 0;
}

template <>
int TosaReference::Tensor5<double>::getTensorValueDouble(const size_t bufLen, double* vals) const
{
    uint32_t idx  = 0;
    int totalVals = 1;

    for (size_t i = 0; i < shape.size(); i++)
    {
        totalVals *= shape[i];
    }

    ASSERT_MSG((size_t)totalVals == bufLen, "Output buffer and tensor size do not match");

    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            for (int i2 = 0; i2 < shape[2]; i2++)
            {
                for (int i3 = 0; i3 < shape[3]; i3++)
                {
                    for (int i4 = 0; i4 < shape[4]; i4++)
                    {
                        vals[idx++] = (*tensor)(i0, i1, i2, i3, i4);
                    }
                }
            }
        }
    }

    return 0;
}

template <>
int TosaReference::Tensor6<double>::getTensorValueDouble(const size_t bufLen, double* vals) const
{
    uint32_t idx  = 0;
    int totalVals = 1;

    for (size_t i = 0; i < shape.size(); i++)
    {
        totalVals *= shape[i];
    }

    ASSERT_MSG((size_t)totalVals == bufLen, "Output buffer and tensor size do not match");

    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            for (int i2 = 0; i2 < shape[2]; i2++)
            {
                for (int i3 = 0; i3 < shape[3]; i3++)
                {
                    for (int i4 = 0; i4 < shape[4]; i4++)
                    {
                        for (int i5 = 0; i5 < shape[5]; i5++)
                        {
                            vals[idx++] = (*tensor)(i0, i1, i2, i3, i4, i5);
                        }
                    }
                }
            }
        }
    }
    return 0;
}

template <class T>
int TosaReference::TensorTemplate<T>::getTensorValueFloat(const size_t bufLen, float* vals) const
{
    FATAL_ERROR("TensorTemplate<T>::getTensorValueFloat should not be called.  "
                "Implement template specialization version.");
    return 0;
}

template <>
int TosaReference::Tensor0<float>::getTensorValueFloat(const size_t bufLen, float* vals) const
{
    int totalVals = 1;

    ASSERT_MSG((size_t)totalVals == bufLen, "Output buffer and tensor size do not match");

    vals[0] = (*tensor)(0);

    return 0;
}

template <>
int TosaReference::Tensor1<float>::getTensorValueFloat(const size_t bufLen, float* vals) const
{
    uint32_t idx  = 0;
    int totalVals = 1;

    for (size_t i = 0; i < shape.size(); i++)
    {
        totalVals *= shape[i];
    }

    ASSERT_MSG((size_t)totalVals == bufLen, "Output buffer and tensor size do not match");

    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        vals[idx++] = (*tensor)(i0);
    }

    return 0;
}

template <>
int TosaReference::Tensor2<float>::getTensorValueFloat(const size_t bufLen, float* vals) const
{
    uint32_t idx  = 0;
    int totalVals = 1;

    for (size_t i = 0; i < shape.size(); i++)
    {
        totalVals *= shape[i];
    }

    ASSERT_MSG((size_t)totalVals == bufLen, "Output buffer and tensor size do not match");

    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            vals[idx++] = (*tensor)(i0, i1);
        }
    }

    return 0;
}

template <>
int TosaReference::Tensor3<float>::getTensorValueFloat(const size_t bufLen, float* vals) const
{
    uint32_t idx  = 0;
    int totalVals = 1;

    for (size_t i = 0; i < shape.size(); i++)
    {
        totalVals *= shape[i];
    }

    ASSERT_MSG((size_t)totalVals == bufLen, "Output buffer and tensor size do not match");

    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            for (int i2 = 0; i2 < shape[2]; i2++)
            {
                vals[idx++] = (*tensor)(i0, i1, i2);
            }
        }
    }

    return 0;
}

template <>
int TosaReference::Tensor4<float>::getTensorValueFloat(const size_t bufLen, float* vals) const
{
    uint32_t idx  = 0;
    int totalVals = 1;

    for (size_t i = 0; i < shape.size(); i++)
    {
        totalVals *= shape[i];
    }

    ASSERT_MSG((size_t)totalVals == bufLen, "Output buffer and tensor size do not match");

    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            for (int i2 = 0; i2 < shape[2]; i2++)
            {
                for (int i3 = 0; i3 < shape[3]; i3++)
                {
                    vals[idx++] = (*tensor)(i0, i1, i2, i3);
                }
            }
        }
    }

    return 0;
}

template <>
int TosaReference::Tensor5<float>::getTensorValueFloat(const size_t bufLen, float* vals) const
{
    uint32_t idx  = 0;
    int totalVals = 1;

    for (size_t i = 0; i < shape.size(); i++)
    {
        totalVals *= shape[i];
    }

    ASSERT_MSG((size_t)totalVals == bufLen, "Output buffer and tensor size do not match");

    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            for (int i2 = 0; i2 < shape[2]; i2++)
            {
                for (int i3 = 0; i3 < shape[3]; i3++)
                {
                    for (int i4 = 0; i4 < shape[4]; i4++)
                    {
                        vals[idx++] = (*tensor)(i0, i1, i2, i3, i4);
                    }
                }
            }
        }
    }

    return 0;
}

template <>
int TosaReference::Tensor6<float>::getTensorValueFloat(const size_t bufLen, float* vals) const
{
    uint32_t idx  = 0;
    int totalVals = 1;

    for (size_t i = 0; i < shape.size(); i++)
    {
        totalVals *= shape[i];
    }

    ASSERT_MSG((size_t)totalVals == bufLen, "Output buffer and tensor size do not match");

    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            for (int i2 = 0; i2 < shape[2]; i2++)
            {
                for (int i3 = 0; i3 < shape[3]; i3++)
                {
                    for (int i4 = 0; i4 < shape[4]; i4++)
                    {
                        for (int i5 = 0; i5 < shape[5]; i5++)
                        {
                            vals[idx++] = (*tensor)(i0, i1, i2, i3, i4, i5);
                        }
                    }
                }
            }
        }
    }
    return 0;
}

template <class T>
int TosaReference::TensorTemplate<T>::getTensorValueInt32(const size_t bufLen, int32_t* vals) const
{
    FATAL_ERROR("TensorTemplate<T>::getTensorValueInt32 should not be called.  "
                "Implement template specialization version.");
    return 0;
}

template <>
int TosaReference::Tensor0<int32_t>::getTensorValueInt32(const size_t bufLen, int32_t* vals) const
{
    int totalVals = 1;

    ASSERT_MSG((size_t)totalVals == bufLen, "Output buffer and tensor size do not match");

    vals[0] = (*tensor)(0);

    return 0;
}

template <>
int TosaReference::Tensor1<int32_t>::getTensorValueInt32(const size_t bufLen, int32_t* vals) const
{
    uint32_t idx  = 0;
    int totalVals = 1;

    for (size_t i = 0; i < shape.size(); i++)
    {
        totalVals *= shape[i];
    }

    ASSERT_MSG((size_t)totalVals == bufLen, "Output buffer and tensor size do not match");

    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        vals[idx++] = (*tensor)(i0);
    }

    return 0;
}

template <>
int TosaReference::Tensor2<int32_t>::getTensorValueInt32(const size_t bufLen, int32_t* vals) const
{
    uint32_t idx  = 0;
    int totalVals = 1;

    for (size_t i = 0; i < shape.size(); i++)
    {
        totalVals *= shape[i];
    }

    ASSERT_MSG((size_t)totalVals == bufLen, "Output buffer and tensor size do not match");

    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            vals[idx++] = (*tensor)(i0, i1);
        }
    }

    return 0;
}

template <>
int TosaReference::Tensor3<int32_t>::getTensorValueInt32(const size_t bufLen, int32_t* vals) const
{
    uint32_t idx  = 0;
    int totalVals = 1;

    for (size_t i = 0; i < shape.size(); i++)
    {
        totalVals *= shape[i];
    }

    ASSERT_MSG((size_t)totalVals == bufLen, "Output buffer and tensor size do not match");

    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            for (int i2 = 0; i2 < shape[2]; i2++)
            {
                vals[idx++] = (*tensor)(i0, i1, i2);
            }
        }
    }

    return 0;
}

template <>
int TosaReference::Tensor4<int32_t>::getTensorValueInt32(const size_t bufLen, int32_t* vals) const
{
    uint32_t idx  = 0;
    int totalVals = 1;

    for (size_t i = 0; i < shape.size(); i++)
    {
        totalVals *= shape[i];
    }

    ASSERT_MSG((size_t)totalVals == bufLen, "Output buffer and tensor size do not match");

    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            for (int i2 = 0; i2 < shape[2]; i2++)
            {
                for (int i3 = 0; i3 < shape[3]; i3++)
                {
                    vals[idx++] = (*tensor)(i0, i1, i2, i3);
                }
            }
        }
    }

    return 0;
}

template <>
int TosaReference::Tensor5<int32_t>::getTensorValueInt32(const size_t bufLen, int32_t* vals) const
{
    uint32_t idx  = 0;
    int totalVals = 1;

    for (size_t i = 0; i < shape.size(); i++)
    {
        totalVals *= shape[i];
    }

    ASSERT_MSG((size_t)totalVals == bufLen, "Output buffer and tensor size do not match");

    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            for (int i2 = 0; i2 < shape[2]; i2++)
            {
                for (int i3 = 0; i3 < shape[3]; i3++)
                {
                    for (int i4 = 0; i4 < shape[4]; i4++)
                    {
                        vals[idx++] = (*tensor)(i0, i1, i2, i3, i4);
                    }
                }
            }
        }
    }

    return 0;
}

template <>
int TosaReference::Tensor6<int32_t>::getTensorValueInt32(const size_t bufLen, int32_t* vals) const
{
    uint32_t idx  = 0;
    int totalVals = 1;

    for (size_t i = 0; i < shape.size(); i++)
    {
        totalVals *= shape[i];
    }

    ASSERT_MSG((size_t)totalVals == bufLen, "Output buffer and tensor size do not match");

    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            for (int i2 = 0; i2 < shape[2]; i2++)
            {
                for (int i3 = 0; i3 < shape[3]; i3++)
                {
                    for (int i4 = 0; i4 < shape[4]; i4++)
                    {
                        for (int i5 = 0; i5 < shape[5]; i5++)
                        {
                            vals[idx++] = (*tensor)(i0, i1, i2, i3, i4, i5);
                        }
                    }
                }
            }
        }
    }
    return 0;
}

template <class T>
int TosaReference::TensorTemplate<T>::getTensorValueInt64(const size_t bufLen, int64_t* vals) const
{
    FATAL_ERROR("TensorTemplate<T>::getTensorValueInt64 should not be called.  "
                "Implement template specialization version.");
    return 0;
}

template <>
int TosaReference::Tensor0<int64_t>::getTensorValueInt64(const size_t bufLen, int64_t* vals) const
{
    int totalVals = 1;

    ASSERT_MSG((size_t)totalVals == bufLen, "Output buffer and tensor size do not match");

    vals[0] = (*tensor)(0);

    return 0;
}

template <>
int TosaReference::Tensor1<int64_t>::getTensorValueInt64(const size_t bufLen, int64_t* vals) const
{
    uint32_t idx  = 0;
    int totalVals = 1;

    for (size_t i = 0; i < shape.size(); i++)
    {
        totalVals *= shape[i];
    }

    ASSERT_MSG((size_t)totalVals == bufLen, "Output buffer and tensor size do not match");

    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        vals[idx++] = (*tensor)(i0);
    }

    return 0;
}

template <>
int TosaReference::Tensor2<int64_t>::getTensorValueInt64(const size_t bufLen, int64_t* vals) const
{
    uint32_t idx  = 0;
    int totalVals = 1;

    for (size_t i = 0; i < shape.size(); i++)
    {
        totalVals *= shape[i];
    }

    ASSERT_MSG((size_t)totalVals == bufLen, "Output buffer and tensor size do not match");

    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            vals[idx++] = (*tensor)(i0, i1);
        }
    }

    return 0;
}

template <>
int TosaReference::Tensor3<int64_t>::getTensorValueInt64(const size_t bufLen, int64_t* vals) const
{
    uint32_t idx  = 0;
    int totalVals = 1;

    for (size_t i = 0; i < shape.size(); i++)
    {
        totalVals *= shape[i];
    }

    ASSERT_MSG((size_t)totalVals == bufLen, "Output buffer and tensor size do not match");

    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            for (int i2 = 0; i2 < shape[2]; i2++)
            {
                vals[idx++] = (*tensor)(i0, i1, i2);
            }
        }
    }

    return 0;
}

template <>
int TosaReference::Tensor4<int64_t>::getTensorValueInt64(const size_t bufLen, int64_t* vals) const
{
    uint32_t idx  = 0;
    int totalVals = 1;

    for (size_t i = 0; i < shape.size(); i++)
    {
        totalVals *= shape[i];
    }

    ASSERT_MSG((size_t)totalVals == bufLen, "Output buffer and tensor size do not match");

    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            for (int i2 = 0; i2 < shape[2]; i2++)
            {
                for (int i3 = 0; i3 < shape[3]; i3++)
                {
                    vals[idx++] = (*tensor)(i0, i1, i2, i3);
                }
            }
        }
    }

    return 0;
}

template <>
int TosaReference::Tensor5<int64_t>::getTensorValueInt64(const size_t bufLen, int64_t* vals) const
{
    uint32_t idx  = 0;
    int totalVals = 1;

    for (size_t i = 0; i < shape.size(); i++)
    {
        totalVals *= shape[i];
    }

    ASSERT_MSG((size_t)totalVals == bufLen, "Output buffer and tensor size do not match");

    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            for (int i2 = 0; i2 < shape[2]; i2++)
            {
                for (int i3 = 0; i3 < shape[3]; i3++)
                {
                    for (int i4 = 0; i4 < shape[4]; i4++)
                    {
                        vals[idx++] = (*tensor)(i0, i1, i2, i3, i4);
                    }
                }
            }
        }
    }

    return 0;
}

template <>
int TosaReference::Tensor6<int64_t>::getTensorValueInt64(const size_t bufLen, int64_t* vals) const
{
    uint32_t idx  = 0;
    int totalVals = 1;

    for (size_t i = 0; i < shape.size(); i++)
    {
        totalVals *= shape[i];
    }

    ASSERT_MSG((size_t)totalVals == bufLen, "Output buffer and tensor size do not match");

    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            for (int i2 = 0; i2 < shape[2]; i2++)
            {
                for (int i3 = 0; i3 < shape[3]; i3++)
                {
                    for (int i4 = 0; i4 < shape[4]; i4++)
                    {
                        for (int i5 = 0; i5 < shape[5]; i5++)
                        {
                            vals[idx++] = (*tensor)(i0, i1, i2, i3, i4, i5);
                        }
                    }
                }
            }
        }
    }
    return 0;
}

template <class T>
int TosaReference::TensorTemplate<T>::getTensorValueBool(const size_t bufLen, bool* vals) const
{
    FATAL_ERROR("TensorTemplate<T>::getTensorValueBool should not be called.  "
                "Implement template specialization version.");
    return 0;
}

template <>
int TosaReference::Tensor0<bool>::getTensorValueBool(const size_t bufLen, bool* vals) const
{
    int totalVals = 1;

    ASSERT_MSG((size_t)totalVals == bufLen, "Output buffer and tensor size do not match");

    vals[0] = (*tensor)(0);

    return 0;
}

template <>
int TosaReference::Tensor1<bool>::getTensorValueBool(const size_t bufLen, bool* vals) const
{
    uint32_t idx  = 0;
    int totalVals = 1;

    for (size_t i = 0; i < shape.size(); i++)
    {
        totalVals *= shape[i];
    }

    ASSERT_MSG((size_t)totalVals == bufLen, "Output buffer and tensor size do not match");

    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        vals[idx++] = (*tensor)(i0);
    }

    return 0;
}

template <>
int TosaReference::Tensor2<bool>::getTensorValueBool(const size_t bufLen, bool* vals) const
{
    uint32_t idx  = 0;
    int totalVals = 1;

    for (size_t i = 0; i < shape.size(); i++)
    {
        totalVals *= shape[i];
    }

    ASSERT_MSG((size_t)totalVals == bufLen, "Output buffer and tensor size do not match");

    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            vals[idx++] = (*tensor)(i0, i1);
        }
    }

    return 0;
}

template <>
int TosaReference::Tensor3<bool>::getTensorValueBool(const size_t bufLen, bool* vals) const
{
    uint32_t idx  = 0;
    int totalVals = 1;

    for (size_t i = 0; i < shape.size(); i++)
    {
        totalVals *= shape[i];
    }

    ASSERT_MSG((size_t)totalVals == bufLen, "Output buffer and tensor size do not match");

    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            for (int i2 = 0; i2 < shape[2]; i2++)
            {
                vals[idx++] = (*tensor)(i0, i1, i2);
            }
        }
    }

    return 0;
}

template <>
int TosaReference::Tensor4<bool>::getTensorValueBool(const size_t bufLen, bool* vals) const
{
    uint32_t idx  = 0;
    int totalVals = 1;

    for (size_t i = 0; i < shape.size(); i++)
    {
        totalVals *= shape[i];
    }

    ASSERT_MSG((size_t)totalVals == bufLen, "Output buffer and tensor size do not match");

    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            for (int i2 = 0; i2 < shape[2]; i2++)
            {
                for (int i3 = 0; i3 < shape[3]; i3++)
                {
                    vals[idx++] = (*tensor)(i0, i1, i2, i3);
                }
            }
        }
    }

    return 0;
}

template <>
int TosaReference::Tensor5<bool>::getTensorValueBool(const size_t bufLen, bool* vals) const
{
    uint32_t idx  = 0;
    int totalVals = 1;

    for (size_t i = 0; i < shape.size(); i++)
    {
        totalVals *= shape[i];
    }

    ASSERT_MSG((size_t)totalVals == bufLen, "Output buffer and tensor size do not match");

    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            for (int i2 = 0; i2 < shape[2]; i2++)
            {
                for (int i3 = 0; i3 < shape[3]; i3++)
                {
                    for (int i4 = 0; i4 < shape[4]; i4++)
                    {
                        vals[idx++] = (*tensor)(i0, i1, i2, i3, i4);
                    }
                }
            }
        }
    }

    return 0;
}

template <>
int TosaReference::Tensor6<bool>::getTensorValueBool(const size_t bufLen, bool* vals) const
{
    uint32_t idx  = 0;
    int totalVals = 1;

    for (size_t i = 0; i < shape.size(); i++)
    {
        totalVals *= shape[i];
    }

    ASSERT_MSG((size_t)totalVals == bufLen, "Output buffer and tensor size do not match");

    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            for (int i2 = 0; i2 < shape[2]; i2++)
            {
                for (int i3 = 0; i3 < shape[3]; i3++)
                {
                    for (int i4 = 0; i4 < shape[4]; i4++)
                    {
                        for (int i5 = 0; i5 < shape[5]; i5++)
                        {
                            vals[idx++] = (*tensor)(i0, i1, i2, i3, i4, i5);
                        }
                    }
                }
            }
        }
    }
    return 0;
}

#define TOSAREF_ZERORANK_TENSOR_ALLOCATE(dtype)                                                                        \
    template <>                                                                                                        \
    int TosaReference::Tensor0<dtype>::allocate()                                                                      \
    {                                                                                                                  \
        ASSERT_MSG(tensor == nullptr, "Error: double allocate Eigen tensor");                                          \
        tensor = new ETensor0<dtype>();                                                                                \
                                                                                                                       \
        if (tensor)                                                                                                    \
            return 0;                                                                                                  \
        else                                                                                                           \
            return 1;                                                                                                  \
    }

#define TOSAREF_TENSOR_ALLOCATE(rank, dtype)                                                                           \
    template <>                                                                                                        \
    int TosaReference::Tensor##rank<dtype>::allocate()                                                                 \
    {                                                                                                                  \
        ASSERT_MSG(tensor == nullptr, "Error: double allocate Eigen tensor");                                          \
        std::array<Eigen::DenseIndex, rank> arrshape;                                                                  \
        std::copy_n(shape.begin(), rank, arrshape.begin());                                                            \
        tensor = new ETensor##rank<dtype>(arrshape);                                                                   \
                                                                                                                       \
        if (tensor)                                                                                                    \
            return 0;                                                                                                  \
        else                                                                                                           \
            return 1;                                                                                                  \
    }

TOSAREF_ZERORANK_TENSOR_ALLOCATE(double)
TOSAREF_TENSOR_ALLOCATE(1, double)
TOSAREF_TENSOR_ALLOCATE(2, double)
TOSAREF_TENSOR_ALLOCATE(3, double)
TOSAREF_TENSOR_ALLOCATE(4, double)
TOSAREF_TENSOR_ALLOCATE(5, double)
TOSAREF_TENSOR_ALLOCATE(6, double)
TOSAREF_ZERORANK_TENSOR_ALLOCATE(float)
TOSAREF_TENSOR_ALLOCATE(1, float)
TOSAREF_TENSOR_ALLOCATE(2, float)
TOSAREF_TENSOR_ALLOCATE(3, float)
TOSAREF_TENSOR_ALLOCATE(4, float)
TOSAREF_TENSOR_ALLOCATE(5, float)
TOSAREF_TENSOR_ALLOCATE(6, float)
TOSAREF_ZERORANK_TENSOR_ALLOCATE(int32_t)
TOSAREF_TENSOR_ALLOCATE(1, int32_t)
TOSAREF_TENSOR_ALLOCATE(2, int32_t)
TOSAREF_TENSOR_ALLOCATE(3, int32_t)
TOSAREF_TENSOR_ALLOCATE(4, int32_t)
TOSAREF_TENSOR_ALLOCATE(5, int32_t)
TOSAREF_TENSOR_ALLOCATE(6, int32_t)
TOSAREF_ZERORANK_TENSOR_ALLOCATE(int64_t)
TOSAREF_TENSOR_ALLOCATE(1, int64_t)
TOSAREF_TENSOR_ALLOCATE(2, int64_t)
TOSAREF_TENSOR_ALLOCATE(3, int64_t)
TOSAREF_TENSOR_ALLOCATE(4, int64_t)
TOSAREF_TENSOR_ALLOCATE(5, int64_t)
TOSAREF_TENSOR_ALLOCATE(6, int64_t)
TOSAREF_ZERORANK_TENSOR_ALLOCATE(bool)
TOSAREF_TENSOR_ALLOCATE(1, bool)
TOSAREF_TENSOR_ALLOCATE(2, bool)
TOSAREF_TENSOR_ALLOCATE(3, bool)
TOSAREF_TENSOR_ALLOCATE(4, bool)
TOSAREF_TENSOR_ALLOCATE(5, bool)
TOSAREF_TENSOR_ALLOCATE(6, bool)

template <>
int TosaReference::Tensor0<double>::dumpTensor(FILE* out) const
{
    char fp_fmt[32];
    snprintf(fp_fmt, sizeof(fp_fmt), "[ %%%sf ]\n", g_func_config.fp_format.c_str());

    if (tensor == nullptr)
    {
        fprintf(out, "<Not allocated>\n");
        return 0;
    }

    fprintf(out, fp_fmt, (*tensor)(0));

    return 0;
}

template <>
int TosaReference::Tensor1<double>::dumpTensor(FILE* out) const
{
    char fp_fmt[32];
    snprintf(fp_fmt, sizeof(fp_fmt), " %%%sf ", g_func_config.fp_format.c_str());

    if (tensor == nullptr)
    {
        fprintf(out, "<Not allocated>\n");
        return 0;
    }

    fprintf(out, "[");
    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        fprintf(out, fp_fmt, (*tensor)(i0));
    }
    fprintf(out, "]\n");

    return 0;
}

template <>
int TosaReference::Tensor2<double>::dumpTensor(FILE* out) const
{
    char fp_fmt[32];
    snprintf(fp_fmt, sizeof(fp_fmt), " %%%sf ", g_func_config.fp_format.c_str());

    if (tensor == nullptr)
    {
        fprintf(out, "<Not allocated>\n");
        return 0;
    }

    fprintf(out, "[");
    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        fprintf(out, "[");
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            fprintf(out, fp_fmt, (*tensor)(i0, i1));
        }
        fprintf(out, "]\n");
    }
    fprintf(out, "]\n");

    return 0;
}

template <>
int TosaReference::Tensor3<double>::dumpTensor(FILE* out) const
{
    char fp_fmt[32];
    snprintf(fp_fmt, sizeof(fp_fmt), " %%%sf ", g_func_config.fp_format.c_str());

    if (tensor == nullptr)
    {
        fprintf(out, "<Not allocated>\n");
        return 0;
    }

    fprintf(out, "[");
    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        fprintf(out, "[");
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            fprintf(out, "[");
            for (int i2 = 0; i2 < shape[2]; i2++)
            {
                fprintf(out, fp_fmt, (*tensor)(i0, i1, i2));
            }
            fprintf(out, "]\n");
        }
        fprintf(out, "]\n");
    }
    fprintf(out, "]\n");

    return 0;
}

template <>
int TosaReference::Tensor4<double>::dumpTensor(FILE* out) const
{
    char fp_fmt[32];
    snprintf(fp_fmt, sizeof(fp_fmt), " %%%sf ", g_func_config.fp_format.c_str());

    if (tensor == nullptr)
    {
        fprintf(out, "<Not allocated>\n");
        return 0;
    }

    fprintf(out, "[");
    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        fprintf(out, "[");
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            fprintf(out, "[");
            for (int i2 = 0; i2 < shape[2]; i2++)
            {
                fprintf(out, "[");
                for (int i3 = 0; i3 < shape[3]; i3++)
                {
                    fprintf(out, fp_fmt, (*tensor)(i0, i1, i2, i3));
                }
                fprintf(out, "]\n");
            }
            fprintf(out, "]\n");
        }
        fprintf(out, "]\n");
    }
    fprintf(out, "]\n");

    return 0;
}

template <>
int TosaReference::Tensor5<double>::dumpTensor(FILE* out) const
{
    char fp_fmt[32];
    snprintf(fp_fmt, sizeof(fp_fmt), " %%%sf ", g_func_config.fp_format.c_str());

    if (tensor == nullptr)
    {
        fprintf(out, "<Not allocated>\n");
        return 0;
    }

    fprintf(out, "[");
    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        fprintf(out, "[");
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            fprintf(out, "[");
            for (int i2 = 0; i2 < shape[2]; i2++)
            {
                fprintf(out, "[");
                for (int i3 = 0; i3 < shape[3]; i3++)
                {
                    fprintf(out, "[");
                    for (int i4 = 0; i4 < shape[4]; i4++)
                    {
                        fprintf(out, fp_fmt, (*tensor)(i0, i1, i2, i3, i4));
                    }
                    fprintf(out, "]\n");
                }
                fprintf(out, "]\n");
            }
            fprintf(out, "]\n");
        }
        fprintf(out, "]\n");
    }
    fprintf(out, "]\n");

    return 0;
}

template <>
int TosaReference::Tensor6<double>::dumpTensor(FILE* out) const
{
    char fp_fmt[32];
    snprintf(fp_fmt, sizeof(fp_fmt), " %%%sf ", g_func_config.fp_format.c_str());

    if (tensor == nullptr)
    {
        fprintf(out, "<Not allocated>\n");
        return 0;
    }

    fprintf(out, "[");
    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        fprintf(out, "[");
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            fprintf(out, "[");
            for (int i2 = 0; i2 < shape[2]; i2++)
            {
                fprintf(out, "[");
                for (int i3 = 0; i3 < shape[3]; i3++)
                {
                    fprintf(out, "[");
                    for (int i4 = 0; i4 < shape[4]; i4++)
                    {
                        fprintf(out, "[");
                        for (int i5 = 0; i5 < shape[5]; i5++)
                        {
                            fprintf(out, fp_fmt, (*tensor)(i0, i1, i2, i3, i4, i5));
                        }
                        fprintf(out, "]\n");
                    }
                    fprintf(out, "]\n");
                }
                fprintf(out, "]\n");
            }
            fprintf(out, "]\n");
        }
        fprintf(out, "]\n");
    }
    fprintf(out, "]\n");

    return 0;
}

template <>
int TosaReference::Tensor0<float>::dumpTensor(FILE* out) const
{
    char fp_fmt[32];
    snprintf(fp_fmt, sizeof(fp_fmt), "[ %%%sf ]\n", g_func_config.fp_format.c_str());

    if (tensor == nullptr)
    {
        fprintf(out, "<Not allocated>\n");
        return 0;
    }

    fprintf(out, fp_fmt, (*tensor)(0));

    return 0;
}

template <>
int TosaReference::Tensor1<float>::dumpTensor(FILE* out) const
{
    char fp_fmt[32];
    snprintf(fp_fmt, sizeof(fp_fmt), " %%%sf ", g_func_config.fp_format.c_str());

    if (tensor == nullptr)
    {
        fprintf(out, "<Not allocated>\n");
        return 0;
    }

    fprintf(out, "[");
    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        fprintf(out, fp_fmt, (*tensor)(i0));
    }
    fprintf(out, "]\n");

    return 0;
}

template <>
int TosaReference::Tensor2<float>::dumpTensor(FILE* out) const
{
    char fp_fmt[32];
    snprintf(fp_fmt, sizeof(fp_fmt), " %%%sf ", g_func_config.fp_format.c_str());

    if (tensor == nullptr)
    {
        fprintf(out, "<Not allocated>\n");
        return 0;
    }

    fprintf(out, "[");
    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        fprintf(out, "[");
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            fprintf(out, fp_fmt, (*tensor)(i0, i1));
        }
        fprintf(out, "]\n");
    }
    fprintf(out, "]\n");

    return 0;
}

template <>
int TosaReference::Tensor3<float>::dumpTensor(FILE* out) const
{
    char fp_fmt[32];
    snprintf(fp_fmt, sizeof(fp_fmt), " %%%sf ", g_func_config.fp_format.c_str());

    if (tensor == nullptr)
    {
        fprintf(out, "<Not allocated>\n");
        return 0;
    }

    fprintf(out, "[");
    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        fprintf(out, "[");
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            fprintf(out, "[");
            for (int i2 = 0; i2 < shape[2]; i2++)
            {
                fprintf(out, fp_fmt, (*tensor)(i0, i1, i2));
            }
            fprintf(out, "]\n");
        }
        fprintf(out, "]\n");
    }
    fprintf(out, "]\n");

    return 0;
}

template <>
int TosaReference::Tensor4<float>::dumpTensor(FILE* out) const
{
    char fp_fmt[32];
    snprintf(fp_fmt, sizeof(fp_fmt), " %%%sf ", g_func_config.fp_format.c_str());

    if (tensor == nullptr)
    {
        fprintf(out, "<Not allocated>\n");
        return 0;
    }

    fprintf(out, "[");
    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        fprintf(out, "[");
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            fprintf(out, "[");
            for (int i2 = 0; i2 < shape[2]; i2++)
            {
                fprintf(out, "[");
                for (int i3 = 0; i3 < shape[3]; i3++)
                {
                    fprintf(out, fp_fmt, (*tensor)(i0, i1, i2, i3));
                }
                fprintf(out, "]\n");
            }
            fprintf(out, "]\n");
        }
        fprintf(out, "]\n");
    }
    fprintf(out, "]\n");

    return 0;
}

template <>
int TosaReference::Tensor5<float>::dumpTensor(FILE* out) const
{
    char fp_fmt[32];
    snprintf(fp_fmt, sizeof(fp_fmt), " %%%sf ", g_func_config.fp_format.c_str());

    if (tensor == nullptr)
    {
        fprintf(out, "<Not allocated>\n");
        return 0;
    }

    fprintf(out, "[");
    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        fprintf(out, "[");
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            fprintf(out, "[");
            for (int i2 = 0; i2 < shape[2]; i2++)
            {
                fprintf(out, "[");
                for (int i3 = 0; i3 < shape[3]; i3++)
                {
                    fprintf(out, "[");
                    for (int i4 = 0; i4 < shape[4]; i4++)
                    {
                        fprintf(out, fp_fmt, (*tensor)(i0, i1, i2, i3, i4));
                    }
                    fprintf(out, "]\n");
                }
                fprintf(out, "]\n");
            }
            fprintf(out, "]\n");
        }
        fprintf(out, "]\n");
    }
    fprintf(out, "]\n");

    return 0;
}

template <>
int TosaReference::Tensor6<float>::dumpTensor(FILE* out) const
{
    char fp_fmt[32];
    snprintf(fp_fmt, sizeof(fp_fmt), " %%%sf ", g_func_config.fp_format.c_str());

    if (tensor == nullptr)
    {
        fprintf(out, "<Not allocated>\n");
        return 0;
    }

    fprintf(out, "[");
    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        fprintf(out, "[");
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            fprintf(out, "[");
            for (int i2 = 0; i2 < shape[2]; i2++)
            {
                fprintf(out, "[");
                for (int i3 = 0; i3 < shape[3]; i3++)
                {
                    fprintf(out, "[");
                    for (int i4 = 0; i4 < shape[4]; i4++)
                    {
                        fprintf(out, "[");
                        for (int i5 = 0; i5 < shape[5]; i5++)
                        {
                            fprintf(out, fp_fmt, (*tensor)(i0, i1, i2, i3, i4, i5));
                        }
                        fprintf(out, "]\n");
                    }
                    fprintf(out, "]\n");
                }
                fprintf(out, "]\n");
            }
            fprintf(out, "]\n");
        }
        fprintf(out, "]\n");
    }
    fprintf(out, "]\n");

    return 0;
}

template <>
int TosaReference::Tensor0<int64_t>::dumpTensor(FILE* out) const
{
    char i64_fmt[32];
    snprintf(i64_fmt, sizeof(i64_fmt), "[ %%ld ]\n");

    if (tensor == nullptr)
    {
        fprintf(out, "<Not allocated>\n");
        return 0;
    }

    fprintf(out, i64_fmt, (*tensor)(0));

    return 0;
}

template <>
int TosaReference::Tensor1<int64_t>::dumpTensor(FILE* out) const
{
    char i64_fmt[32];
    snprintf(i64_fmt, sizeof(i64_fmt), " %%ld ");

    if (tensor == nullptr)
    {
        fprintf(out, "<Not allocated>\n");
        return 0;
    }

    fprintf(out, "[");
    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        fprintf(out, i64_fmt, (*tensor)(i0));
    }
    fprintf(out, "]\n");

    return 0;
}

template <>
int TosaReference::Tensor2<int64_t>::dumpTensor(FILE* out) const
{
    char i64_fmt[32];
    snprintf(i64_fmt, sizeof(i64_fmt), " %%ld ");

    if (tensor == nullptr)
    {
        fprintf(out, "<Not allocated>\n");
        return 0;
    }

    fprintf(out, "[");
    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        fprintf(out, "[");
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            fprintf(out, i64_fmt, (*tensor)(i0, i1));
        }
        fprintf(out, "]\n");
    }
    fprintf(out, "]\n");

    return 0;
}

template <>
int TosaReference::Tensor3<int64_t>::dumpTensor(FILE* out) const
{
    char i64_fmt[32];
    snprintf(i64_fmt, sizeof(i64_fmt), " %%ld ");

    if (tensor == nullptr)
    {
        fprintf(out, "<Not allocated>\n");
        return 0;
    }

    fprintf(out, "[");
    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        fprintf(out, "[");
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            fprintf(out, "[");
            for (int i2 = 0; i2 < shape[2]; i2++)
            {
                fprintf(out, i64_fmt, (*tensor)(i0, i1, i2));
            }
            fprintf(out, "]\n");
        }
        fprintf(out, "]\n");
    }
    fprintf(out, "]\n");

    return 0;
}

template <>
int TosaReference::Tensor4<int64_t>::dumpTensor(FILE* out) const
{
    char i64_fmt[32];
    snprintf(i64_fmt, sizeof(i64_fmt), " %%ld ");

    if (tensor == nullptr)
    {
        fprintf(out, "<Not allocated>\n");
        return 0;
    }

    fprintf(out, "[");
    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        fprintf(out, "[");
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            fprintf(out, "[");
            for (int i2 = 0; i2 < shape[2]; i2++)
            {
                fprintf(out, "[");
                for (int i3 = 0; i3 < shape[3]; i3++)
                {
                    fprintf(out, i64_fmt, (*tensor)(i0, i1, i2, i3));
                }
                fprintf(out, "]\n");
            }
            fprintf(out, "]\n");
        }
        fprintf(out, "]\n");
    }
    fprintf(out, "]\n");

    return 0;
}

template <>
int TosaReference::Tensor5<int64_t>::dumpTensor(FILE* out) const
{
    char i64_fmt[32];
    snprintf(i64_fmt, sizeof(i64_fmt), " %%ld ");

    if (tensor == nullptr)
    {
        fprintf(out, "<Not allocated>\n");
        return 0;
    }

    fprintf(out, "[");
    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        fprintf(out, "[");
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            fprintf(out, "[");
            for (int i2 = 0; i2 < shape[2]; i2++)
            {
                fprintf(out, "[");
                for (int i3 = 0; i3 < shape[3]; i3++)
                {
                    fprintf(out, "[");
                    for (int i4 = 0; i4 < shape[4]; i4++)
                    {
                        fprintf(out, i64_fmt, (*tensor)(i0, i1, i2, i3, i4));
                    }
                    fprintf(out, "]\n");
                }
                fprintf(out, "]\n");
            }
            fprintf(out, "]\n");
        }
        fprintf(out, "]\n");
    }
    fprintf(out, "]\n");

    return 0;
}

template <>
int TosaReference::Tensor6<int64_t>::dumpTensor(FILE* out) const
{
    char i64_fmt[32];
    snprintf(i64_fmt, sizeof(i64_fmt), " %%ld ");

    if (tensor == nullptr)
    {
        fprintf(out, "<Not allocated>\n");
        return 0;
    }

    fprintf(out, "[");
    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        fprintf(out, "[");
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            fprintf(out, "[");
            for (int i2 = 0; i2 < shape[2]; i2++)
            {
                fprintf(out, "[");
                for (int i3 = 0; i3 < shape[3]; i3++)
                {
                    fprintf(out, "[");
                    for (int i4 = 0; i4 < shape[4]; i4++)
                    {
                        fprintf(out, "[");
                        for (int i5 = 0; i5 < shape[5]; i5++)
                        {
                            fprintf(out, i64_fmt, (*tensor)(i0, i1, i2, i3, i4, i5));
                        }
                        fprintf(out, "]\n");
                    }
                    fprintf(out, "]\n");
                }
                fprintf(out, "]\n");
            }
            fprintf(out, "]\n");
        }
        fprintf(out, "]\n");
    }
    fprintf(out, "]\n");

    return 0;
}

template <>
int TosaReference::Tensor0<int32_t>::dumpTensor(FILE* out) const
{
    char i32_fmt[32];
    snprintf(i32_fmt, sizeof(i32_fmt), "[ %%d ]\n");

    if (tensor == nullptr)
    {
        fprintf(out, "<Not allocated>\n");
        return 0;
    }

    fprintf(out, i32_fmt, (*tensor)(0));

    return 0;
}

template <>
int TosaReference::Tensor1<int32_t>::dumpTensor(FILE* out) const
{
    char i32_fmt[32];
    snprintf(i32_fmt, sizeof(i32_fmt), " %%d ");

    if (tensor == nullptr)
    {
        fprintf(out, "<Not allocated>\n");
        return 0;
    }

    fprintf(out, "[");
    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        fprintf(out, i32_fmt, (*tensor)(i0));
    }
    fprintf(out, "]\n");

    return 0;
}

template <>
int TosaReference::Tensor2<int32_t>::dumpTensor(FILE* out) const
{
    char i32_fmt[32];
    snprintf(i32_fmt, sizeof(i32_fmt), " %%d ");

    if (tensor == nullptr)
    {
        fprintf(out, "<Not allocated>\n");
        return 0;
    }

    if (tensor == nullptr)
    {
        fprintf(out, "<Not allocated>\n");
        return 0;
    }

    fprintf(out, "[");
    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        fprintf(out, "[");
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            fprintf(out, i32_fmt, (*tensor)(i0, i1));
        }
        fprintf(out, "]\n");
    }
    fprintf(out, "]\n");

    return 0;
}

template <>
int TosaReference::Tensor3<int32_t>::dumpTensor(FILE* out) const
{
    char i32_fmt[32];
    snprintf(i32_fmt, sizeof(i32_fmt), " %%d ");

    if (tensor == nullptr)
    {
        fprintf(out, "<Not allocated>\n");
        return 0;
    }

    if (tensor == nullptr)
    {
        fprintf(out, "<Not allocated>\n");
        return 0;
    }

    fprintf(out, "[");
    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        fprintf(out, "[");
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            fprintf(out, "[");
            for (int i2 = 0; i2 < shape[2]; i2++)
            {
                fprintf(out, i32_fmt, (*tensor)(i0, i1, i2));
            }
            fprintf(out, "]\n");
        }
        fprintf(out, "]\n");
    }
    fprintf(out, "]\n");

    return 0;
}

template <>
int TosaReference::Tensor4<int32_t>::dumpTensor(FILE* out) const
{
    char i32_fmt[32];
    snprintf(i32_fmt, sizeof(i32_fmt), " %%d ");

    if (tensor == nullptr)
    {
        fprintf(out, "<Not allocated>\n");
        return 0;
    }

    fprintf(out, "[");
    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        fprintf(out, "[");
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            fprintf(out, "[");
            for (int i2 = 0; i2 < shape[2]; i2++)
            {
                fprintf(out, "[");
                for (int i3 = 0; i3 < shape[3]; i3++)
                {
                    fprintf(out, i32_fmt, (*tensor)(i0, i1, i2, i3));
                }
                fprintf(out, "]\n");
            }
            fprintf(out, "]\n");
        }
        fprintf(out, "]\n");
    }
    fprintf(out, "]\n");

    return 0;
}

template <>
int TosaReference::Tensor5<int32_t>::dumpTensor(FILE* out) const
{
    char i32_fmt[32];
    snprintf(i32_fmt, sizeof(i32_fmt), " %%d ");

    if (tensor == nullptr)
    {
        fprintf(out, "<Not allocated>\n");
        return 0;
    }

    fprintf(out, "[");
    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        fprintf(out, "[");
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            fprintf(out, "[");
            for (int i2 = 0; i2 < shape[2]; i2++)
            {
                fprintf(out, "[");
                for (int i3 = 0; i3 < shape[3]; i3++)
                {
                    fprintf(out, "[");
                    for (int i4 = 0; i4 < shape[4]; i4++)
                    {
                        fprintf(out, i32_fmt, (*tensor)(i0, i1, i2, i3, i4));
                    }
                    fprintf(out, "]\n");
                }
                fprintf(out, "]\n");
            }
            fprintf(out, "]\n");
        }
        fprintf(out, "]\n");
    }
    fprintf(out, "]\n");

    return 0;
}

template <>
int TosaReference::Tensor6<int32_t>::dumpTensor(FILE* out) const
{
    char i32_fmt[32];
    snprintf(i32_fmt, sizeof(i32_fmt), " %%d ");

    if (tensor == nullptr)
    {
        fprintf(out, "<Not allocated>\n");
        return 0;
    }

    fprintf(out, "[");
    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        fprintf(out, "[");
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            fprintf(out, "[");
            for (int i2 = 0; i2 < shape[2]; i2++)
            {
                fprintf(out, "[");
                for (int i3 = 0; i3 < shape[3]; i3++)
                {
                    fprintf(out, "[");
                    for (int i4 = 0; i4 < shape[4]; i4++)
                    {
                        fprintf(out, "[");
                        for (int i5 = 0; i5 < shape[5]; i5++)
                        {
                            fprintf(out, i32_fmt, (*tensor)(i0, i1, i2, i3, i4, i5));
                        }
                        fprintf(out, "]\n");
                    }
                    fprintf(out, "]\n");
                }
                fprintf(out, "]\n");
            }
            fprintf(out, "]\n");
        }
        fprintf(out, "]\n");
    }
    fprintf(out, "]\n");

    return 0;
}

template <>
int TosaReference::Tensor0<bool>::dumpTensor(FILE* out) const
{
    char bool_fmt[32];
    snprintf(bool_fmt, sizeof(bool_fmt), "[ %%s ]\n");

    if (tensor == nullptr)
    {
        fprintf(out, "<Not allocated>\n");
        return 0;
    }

    fprintf(out, bool_fmt, bool_to_str((*tensor)(0)));

    return 0;
}

template <>
int TosaReference::Tensor1<bool>::dumpTensor(FILE* out) const
{
    char bool_fmt[32];
    snprintf(bool_fmt, sizeof(bool_fmt), " %%s ");

    if (tensor == nullptr)
    {
        fprintf(out, "<Not allocated>\n");
        return 0;
    }

    fprintf(out, "[");
    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        fprintf(out, bool_fmt, bool_to_str((*tensor)(i0)));
    }
    fprintf(out, "]\n");

    return 0;
}

template <>
int TosaReference::Tensor2<bool>::dumpTensor(FILE* out) const
{
    char bool_fmt[32];
    snprintf(bool_fmt, sizeof(bool_fmt), " %%s ");

    if (tensor == nullptr)
    {
        fprintf(out, "<Not allocated>\n");
        return 0;
    }

    fprintf(out, "[");
    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        fprintf(out, "[");
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            fprintf(out, bool_fmt, bool_to_str((*tensor)(i0, i1)));
        }
        fprintf(out, "]\n");
    }
    fprintf(out, "]\n");

    return 0;
}

template <>
int TosaReference::Tensor3<bool>::dumpTensor(FILE* out) const
{
    char bool_fmt[32];
    snprintf(bool_fmt, sizeof(bool_fmt), " %%s ");

    if (tensor == nullptr)
    {
        fprintf(out, "<Not allocated>\n");
        return 0;
    }

    fprintf(out, "[");
    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        fprintf(out, "[");
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            fprintf(out, "[");
            for (int i2 = 0; i2 < shape[2]; i2++)
            {
                fprintf(out, bool_fmt, bool_to_str((*tensor)(i0, i1, i2)));
            }
            fprintf(out, "]\n");
        }
        fprintf(out, "]\n");
    }
    fprintf(out, "]\n");

    return 0;
}

template <>
int TosaReference::Tensor4<bool>::dumpTensor(FILE* out) const
{
    char bool_fmt[32];
    snprintf(bool_fmt, sizeof(bool_fmt), " %%s ");

    if (tensor == nullptr)
    {
        fprintf(out, "<Not allocated>\n");
        return 0;
    }

    fprintf(out, "[");
    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        fprintf(out, "[");
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            fprintf(out, "[");
            for (int i2 = 0; i2 < shape[2]; i2++)
            {
                fprintf(out, "[");
                for (int i3 = 0; i3 < shape[3]; i3++)
                {
                    fprintf(out, bool_fmt, bool_to_str((*tensor)(i0, i1, i2, i3)));
                }
                fprintf(out, "]\n");
            }
            fprintf(out, "]\n");
        }
        fprintf(out, "]\n");
    }
    fprintf(out, "]\n");

    return 0;
}

template <>
int TosaReference::Tensor5<bool>::dumpTensor(FILE* out) const
{
    char bool_fmt[32];
    snprintf(bool_fmt, sizeof(bool_fmt), " %%s ");

    if (tensor == nullptr)
    {
        fprintf(out, "<Not allocated>\n");
        return 0;
    }

    fprintf(out, "[");
    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        fprintf(out, "[");
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            fprintf(out, "[");
            for (int i2 = 0; i2 < shape[2]; i2++)
            {
                fprintf(out, "[");
                for (int i3 = 0; i3 < shape[3]; i3++)
                {
                    fprintf(out, "[");
                    for (int i4 = 0; i4 < shape[4]; i4++)
                    {
                        fprintf(out, bool_fmt, bool_to_str((*tensor)(i0, i1, i2, i3, i4)));
                    }
                    fprintf(out, "]\n");
                }
                fprintf(out, "]\n");
            }
            fprintf(out, "]\n");
        }
        fprintf(out, "]\n");
    }
    fprintf(out, "]\n");

    return 0;
}

template <>
int TosaReference::Tensor6<bool>::dumpTensor(FILE* out) const
{
    char bool_fmt[32];
    snprintf(bool_fmt, sizeof(bool_fmt), " %%s ");

    if (tensor == nullptr)
    {
        fprintf(out, "<Not allocated>\n");
        return 0;
    }

    fprintf(out, "[");
    for (int i0 = 0; i0 < shape[0]; i0++)
    {
        fprintf(out, "[");
        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            fprintf(out, "[");
            for (int i2 = 0; i2 < shape[2]; i2++)
            {
                fprintf(out, "[");
                for (int i3 = 0; i3 < shape[3]; i3++)
                {
                    fprintf(out, "[");
                    for (int i4 = 0; i4 < shape[4]; i4++)
                    {
                        fprintf(out, "[");
                        for (int i5 = 0; i5 < shape[5]; i5++)
                        {
                            fprintf(out, bool_fmt, bool_to_str((*tensor)(i0, i1, i2, i3, i4, i5)));
                        }
                        fprintf(out, "]\n");
                    }
                    fprintf(out, "]\n");
                }
                fprintf(out, "]\n");
            }
            fprintf(out, "]\n");
        }
        fprintf(out, "]\n");
    }
    fprintf(out, "]\n");

    return 0;
}

template <class T>
int TosaReference::TensorTemplate<T>::dumpTensor(FILE* out) const
{
    return 0;
}

// template explicit specialization
template class TosaReference::TensorTemplate<Eigen::Tensor<double, 0>>;
template class TosaReference::TensorTemplate<Eigen::Tensor<double, 1>>;
template class TosaReference::TensorTemplate<Eigen::Tensor<double, 2>>;
template class TosaReference::TensorTemplate<Eigen::Tensor<double, 3>>;
template class TosaReference::TensorTemplate<Eigen::Tensor<double, 4>>;
template class TosaReference::TensorTemplate<Eigen::Tensor<double, 5>>;
template class TosaReference::TensorTemplate<Eigen::Tensor<double, 6>>;

template class TosaReference::TensorTemplate<Eigen::Tensor<float, 0>>;
template class TosaReference::TensorTemplate<Eigen::Tensor<float, 1>>;
template class TosaReference::TensorTemplate<Eigen::Tensor<float, 2>>;
template class TosaReference::TensorTemplate<Eigen::Tensor<float, 3>>;
template class TosaReference::TensorTemplate<Eigen::Tensor<float, 4>>;
template class TosaReference::TensorTemplate<Eigen::Tensor<float, 5>>;
template class TosaReference::TensorTemplate<Eigen::Tensor<float, 6>>;

template class TosaReference::TensorTemplate<Eigen::Tensor<int32_t, 0>>;
template class TosaReference::TensorTemplate<Eigen::Tensor<int32_t, 1>>;
template class TosaReference::TensorTemplate<Eigen::Tensor<int32_t, 2>>;
template class TosaReference::TensorTemplate<Eigen::Tensor<int32_t, 3>>;
template class TosaReference::TensorTemplate<Eigen::Tensor<int32_t, 4>>;
template class TosaReference::TensorTemplate<Eigen::Tensor<int32_t, 5>>;
template class TosaReference::TensorTemplate<Eigen::Tensor<int32_t, 6>>;

template class TosaReference::TensorTemplate<Eigen::Tensor<int64_t, 0>>;
template class TosaReference::TensorTemplate<Eigen::Tensor<int64_t, 1>>;
template class TosaReference::TensorTemplate<Eigen::Tensor<int64_t, 2>>;
template class TosaReference::TensorTemplate<Eigen::Tensor<int64_t, 3>>;
template class TosaReference::TensorTemplate<Eigen::Tensor<int64_t, 4>>;
template class TosaReference::TensorTemplate<Eigen::Tensor<int64_t, 5>>;
template class TosaReference::TensorTemplate<Eigen::Tensor<int64_t, 6>>;

template class TosaReference::TensorTemplate<Eigen::Tensor<bool, 0>>;
template class TosaReference::TensorTemplate<Eigen::Tensor<bool, 1>>;
template class TosaReference::TensorTemplate<Eigen::Tensor<bool, 2>>;
template class TosaReference::TensorTemplate<Eigen::Tensor<bool, 3>>;
template class TosaReference::TensorTemplate<Eigen::Tensor<bool, 4>>;
template class TosaReference::TensorTemplate<Eigen::Tensor<bool, 5>>;
template class TosaReference::TensorTemplate<Eigen::Tensor<bool, 6>>;

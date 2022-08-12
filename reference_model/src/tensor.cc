
// Copyright (c) 2020-2022, ARM Limited.
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
#include "half.hpp"

using namespace TosaReference;
using namespace Eigen;
using namespace tosa;

TosaReference::Tensor::Tensor(std::string tensorName_, DType tensorDtype_, std::vector<int> shape_)
{
    tensorName  = std::string(tensorName_);
    tensorDtype = tensorDtype_;
    shape       = std::vector<int>(shape_);
    producer    = nullptr;
    isValid     = false;
    consumers.clear();
    isSubgraphInput  = false;
    isSubgraphOutput = false;
}

TosaReference::Tensor::~Tensor()
{}

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
    fprintf(out, "Name: %s DType=%s isValid=%d Rank=%d Shape=%s\n", tensorName.c_str(), EnumNamesDType()[getDtype()],
            getIsValid(), getRank(), getShapeAsString().c_str());

    return 0;
}

int TosaReference::Tensor::dumpTensorParams(std::ostream& out) const
{
    out << "Name: " << getName() << " DType=" << EnumNamesDType()[getDtype()] << " isValid=" << getIsValid()
        << " Rank=" << getRank() << " Shape=" << getShapeAsString() << "\n";

    return 0;
}

int TosaReference::Tensor::readFromNpyFile(const char* filename)
{
    uint32_t elements   = getElementCount();
    float* fdatabuf     = nullptr;
    half_float::half* f16databuf = nullptr;
    int32_t* i32databuf = nullptr;
    int64_t* i64databuf = nullptr;
    bool* bdatabuf      = nullptr;
    NumpyUtilities::NPError nperror;

    switch (getDtype())
    {
        case DType_FLOAT:
            fdatabuf = (float*)calloc(sizeof(float), elements);
            ASSERT_MEM(fdatabuf);

            nperror = NumpyUtilities::readFromNpyFile(filename, elements, fdatabuf);
            break;
        case DType_FP16:
            f16databuf = (half_float::half*)calloc(sizeof(half_float::half), elements);
            ASSERT_MEM(f16databuf);
            fdatabuf = (float*)calloc(sizeof(float), elements);
            ASSERT_MEM(fdatabuf);

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
            FATAL_ERROR("unsupported tensor type=%s", EnumNamesDType()[getDtype()]);
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
                        EnumNamesDType()[getDtype()], getName().c_str(), filename);
        case NumpyUtilities::HEADER_PARSE_ERROR:
            FATAL_ERROR("Numpy header parsing error for file: %s", filename);
        case NumpyUtilities::BUFFER_SIZE_MISMATCH:
            FATAL_ERROR("Buffer size does not match numpy file size for tensor %s filename %s", getName().c_str(),
                        filename);
        default:
            FATAL_ERROR("Unknown error parsing Numpy file: %s", filename);
    }

    switch (getDtype())
    {
        case DType_FP16:
            // Convert from fp16 to fp32
            for (uint32_t i=0; i < elements; i++) {
                fdatabuf[i] = half_float::half_cast<float, half_float::half>(f16databuf[i]);
            }
            // Fall through to DType_FLOAT case
        case DType_FLOAT:
            if (setTensorValueFloat(elements, fdatabuf))
            {
                if (f16databuf)
                    free(f16databuf);
                free(fdatabuf);
                return 1;
            }
            break;
        case DType_INT32:
        case DType_UINT8:
        case DType_INT4:
        case DType_INT8:
        case DType_INT16:
        case DType_UINT16:
            if (setTensorValueInt32(elements, i32databuf))
            {
                free(i32databuf);
                return 1;
            }
            break;
        case DType_INT48:
            if (setTensorValueInt64(elements, i64databuf))
            {
                free(i64databuf);
                return 1;
            }
            break;
        case DType_BOOL:
            if (setTensorValueBool(elements, bdatabuf))
            {
                free(i32databuf);
                return 1;
            }
            break;
        default:
            FATAL_ERROR("unsupported tensor type=%s", EnumNamesDType()[getDtype()]);
    }

    setIsValid();

    if (fdatabuf)
        free(fdatabuf);
    if (f16databuf)
        free(f16databuf);
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
    float* fdatabuf     = nullptr;
    half_float::half* f16databuf  = nullptr;
    int32_t* i32databuf = nullptr;
    int64_t* i64databuf = nullptr;
    bool* bdatabuf      = nullptr;
    NumpyUtilities::NPError nperror;
    uint32_t elements = getElementCount();

    switch (getDtype())
    {
        case DType_FLOAT:
            fdatabuf = (float*)calloc(sizeof(float), elements);
            ASSERT_MEM(fdatabuf);

            if (getTensorValueFloat(elements, fdatabuf))
            {
                free(fdatabuf);
                return 1;
            }

            nperror = NumpyUtilities::writeToNpyFile(filename, shape, fdatabuf);

            free(fdatabuf);
            break;
        case DType_FP16:
            fdatabuf = (float*)calloc(sizeof(float), elements);
            ASSERT_MEM(fdatabuf);
            f16databuf = (half_float::half*)calloc(sizeof(half_float::half), elements);
            ASSERT_MEM(f16databuf);

            if (getTensorValueFloat(elements, fdatabuf))
            {
                free(fdatabuf);
                free(f16databuf);
                return 1;
            }
            // Convert fp32 to fp16
            for (uint32_t i=0; i < elements; i++) {
                f16databuf[i] = half_float::half_cast<half_float::half, float>(fdatabuf[i]);
            }
            nperror = NumpyUtilities::writeToNpyFile(filename, shape, f16databuf);

            free(fdatabuf);
            free(f16databuf);
            break;
        case DType_INT32:
        case DType_UINT8:
        case DType_INT4:
        case DType_INT8:
        case DType_INT16:
        case DType_UINT16:
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
        case DType_INT48:
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
        case DType_BOOL:
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
        default:
            FATAL_ERROR("unsupported tensor type=%s", EnumNamesDType()[getDtype()]);
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
        uint32_t src_rank = src->getRank();                                                                            \
        uint32_t dst_rank = this->getRank();                                                                           \
        DType src_dtype   = src->getDtype();                                                                           \
        DType dst_dtype   = this->getDtype();                                                                          \
        bool tensor_match = true;                                                                                      \
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
                    src->getName().c_str(), src_rank, EnumNamesDType()[src_dtype], src->getShapeAsString().c_str(),    \
                    this->getName().c_str(), dst_rank, EnumNamesDType()[dst_dtype], this->getShapeAsString().c_str()); \
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

int TosaReference::Tensor::readfromVector(const std::vector<float>& vals)
{
    uint32_t elements = getElementCount();
    switch (getDtype())
    {
        case DType_FLOAT:
            if (vals.size() != elements)
            {
                WARNING("The input size (%ld) doesn't match the number of elements (%d) assigned to the tensor.",
                        vals.size(), elements);
                return -1;
            }

            setTensorValueFloat(elements, vals.data());
            break;
        default:
            WARNING("The input type (float) doesn't match the data type assigned to the tensor (%s).",
                    EnumNameDType(getDtype()));
            return -2;
    }
    setIsValid();
    return 0;
}

int TosaReference::Tensor::readfromVector(const std::vector<int32_t>& vals)
{
    uint32_t elements = getElementCount();
    switch (getDtype())
    {
        case DType_INT32:
        case DType_UINT8:
        case DType_INT4:
        case DType_INT8:
        case DType_INT16:
        case DType_UINT16:
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
                    EnumNameDType(getDtype()));
            return -2;
    }
    setIsValid();
    return 0;
}

int TosaReference::Tensor::readfromVector(const std::vector<int64_t>& vals)
{
    uint32_t elements = getElementCount();
    switch (getDtype())
    {
        case DType_INT48:
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
                    EnumNameDType(getDtype()));
            return -2;
    }
    setIsValid();
    return 0;
}

int TosaReference::Tensor::readfromVector(const std::vector<unsigned char>& vals)
{
    uint32_t elements = getElementCount();

    switch (getDtype())
    {
        case DType_BOOL:
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
                    EnumNameDType(getDtype()));
            return -2;
    }
    setIsValid();
    return 0;
}

int TosaReference::Tensor::writeToVector(std::vector<float>& vals)
{
    uint32_t elements = getElementCount();

    switch (getDtype())
    {
        case DType_FLOAT:
            if (vals.size() != elements)
            {
                WARNING("The output size (%ld) doesn't match the number of elements (%d) assigned to the tensor.",
                        vals.size(), elements);
                return -1;
            }

            getTensorValueFloat(elements, vals.data());
            break;
        default:
            WARNING("The output type (float) doesn't match the data type assigned to the tensor (%s).",
                    EnumNameDType(getDtype()));
            return -2;
    }
    return 0;
}

int TosaReference::Tensor::writeToVector(std::vector<int32_t>& vals)
{
    uint32_t elements = getElementCount();

    switch (getDtype())
    {
        case DType_INT32:
        case DType_UINT8:
        case DType_INT4:
        case DType_INT8:
        case DType_INT16:
        case DType_UINT16:
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
                    EnumNameDType(getDtype()));
            return -2;
    }
    return 0;
}

int TosaReference::Tensor::writeToVector(std::vector<int64_t>& vals)
{
    uint32_t elements = getElementCount();

    switch (getDtype())
    {
        case tosa::DType_INT48:
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
                    EnumNameDType(getDtype()));
            return -2;
    }
    return 0;
}

int TosaReference::Tensor::writeToVector(std::vector<unsigned char>& vals)
{
    uint32_t elements = getElementCount();

    switch (getDtype())
    {
        case tosa::DType_BOOL:
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
                    EnumNameDType(getDtype()));
            return -2;
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

template <>
int TosaReference::Tensor0<float>::allocate()
{
    ASSERT_MSG(tensor == nullptr, "Error: double allocate Eigen tensor");
    tensor = new ETensor0<float>();

    if (tensor)
        return 0;
    else
        return 1;
}
template <>
int TosaReference::Tensor1<float>::allocate()
{
    ASSERT_MSG(tensor == nullptr, "Error: double allocate Eigen tensor");
    tensor = new ETensor1<float>(shape[0]);
    if (tensor)
        return 0;
    else
        return 1;
}
template <>
int TosaReference::Tensor2<float>::allocate()
{
    ASSERT_MSG(tensor == nullptr, "Error: double allocate Eigen tensor");
    tensor = new ETensor2<float>(shape[0], shape[1]);
    if (tensor)
        return 0;
    else
        return 1;
}

template <>
int TosaReference::Tensor3<float>::allocate()
{
    ASSERT_MSG(tensor == nullptr, "Error: double allocate Eigen tensor");
    tensor = new ETensor3<float>(shape[0], shape[1], shape[2]);
    if (tensor)
        return 0;
    else
        return 1;
}

template <>
int TosaReference::Tensor4<float>::allocate()
{
    ASSERT_MSG(tensor == nullptr, "Error: double allocate Eigen tensor");
    tensor = new ETensor4<float>(shape[0], shape[1], shape[2], shape[3]);
    if (tensor)
        return 0;
    else
        return 1;
}

template <>
int TosaReference::Tensor5<float>::allocate()
{
    ASSERT_MSG(tensor == nullptr, "Error: double allocate Eigen tensor");
    tensor = new ETensor5<float>(shape[0], shape[1], shape[2], shape[3], shape[4]);
    if (tensor)
        return 0;
    else
        return 1;
}

template <>
int TosaReference::Tensor6<float>::allocate()
{
    ASSERT_MSG(tensor == nullptr, "Error: double allocate Eigen tensor");
    tensor = new ETensor6<float>(shape[0], shape[1], shape[2], shape[3], shape[4], shape[5]);
    if (tensor)
        return 0;
    else
        return 1;
}

template <>
int TosaReference::Tensor0<int32_t>::allocate()
{
    ASSERT_MSG(tensor == nullptr, "Error: double allocate Eigen tensor");
    tensor = new ETensor0<int32_t>();
    if (tensor)
        return 0;
    else
        return 1;
}
template <>
int TosaReference::Tensor1<int32_t>::allocate()
{
    ASSERT_MSG(tensor == nullptr, "Error: double allocate Eigen tensor");
    tensor = new ETensor1<int32_t>(shape[0]);
    if (tensor)
        return 0;
    else
        return 1;
}
template <>
int TosaReference::Tensor2<int32_t>::allocate()
{
    ASSERT_MSG(tensor == nullptr, "Error: double allocate Eigen tensor");
    tensor = new ETensor2<int32_t>(shape[0], shape[1]);
    if (tensor)
        return 0;
    else
        return 1;
}

template <>
int TosaReference::Tensor3<int32_t>::allocate()
{
    ASSERT_MSG(tensor == nullptr, "Error: double allocate Eigen tensor");
    tensor = new ETensor3<int32_t>(shape[0], shape[1], shape[2]);
    if (tensor)
        return 0;
    else
        return 1;
}

template <>
int TosaReference::Tensor4<int32_t>::allocate()
{
    ASSERT_MSG(tensor == nullptr, "Error: double allocate Eigen tensor");
    tensor = new ETensor4<int32_t>(shape[0], shape[1], shape[2], shape[3]);
    if (tensor)
        return 0;
    else
        return 1;
}

template <>
int TosaReference::Tensor5<int32_t>::allocate()
{
    ASSERT_MSG(tensor == nullptr, "Error: double allocate Eigen tensor");
    tensor = new ETensor5<int32_t>(shape[0], shape[1], shape[2], shape[3], shape[4]);
    if (tensor)
        return 0;
    else
        return 1;
}

template <>
int TosaReference::Tensor6<int32_t>::allocate()
{
    ASSERT_MSG(tensor == nullptr, "Error: double allocate Eigen tensor");
    tensor = new ETensor6<int32_t>(shape[0], shape[1], shape[2], shape[3], shape[4], shape[5]);
    if (tensor)
        return 0;
    else
        return 1;
}

template <>
int TosaReference::Tensor0<int64_t>::allocate()
{
    ASSERT_MSG(tensor == nullptr, "Error: double allocate Eigen tensor");
    tensor = new ETensor0<int64_t>();
    if (tensor)
        return 0;
    else
        return 1;
}
template <>
int TosaReference::Tensor1<int64_t>::allocate()
{
    ASSERT_MSG(tensor == nullptr, "Error: double allocate Eigen tensor");
    tensor = new ETensor1<int64_t>(shape[0]);
    if (tensor)
        return 0;
    else
        return 1;
}
template <>
int TosaReference::Tensor2<int64_t>::allocate()
{
    ASSERT_MSG(tensor == nullptr, "Error: double allocate Eigen tensor");
    tensor = new ETensor2<int64_t>(shape[0], shape[1]);
    if (tensor)
        return 0;
    else
        return 1;
}

template <>
int TosaReference::Tensor3<int64_t>::allocate()
{
    ASSERT_MSG(tensor == nullptr, "Error: double allocate Eigen tensor");
    tensor = new ETensor3<int64_t>(shape[0], shape[1], shape[2]);
    if (tensor)
        return 0;
    else
        return 1;
}

template <>
int TosaReference::Tensor4<int64_t>::allocate()
{
    ASSERT_MSG(tensor == nullptr, "Error: double allocate Eigen tensor");
    tensor = new ETensor4<int64_t>(shape[0], shape[1], shape[2], shape[3]);
    if (tensor)
        return 0;
    else
        return 1;
}

template <>
int TosaReference::Tensor5<int64_t>::allocate()
{
    ASSERT_MSG(tensor == nullptr, "Error: double allocate Eigen tensor");
    tensor = new ETensor5<int64_t>(shape[0], shape[1], shape[2], shape[3], shape[4]);
    if (tensor)
        return 0;
    else
        return 1;
}

template <>
int TosaReference::Tensor6<int64_t>::allocate()
{
    ASSERT_MSG(tensor == nullptr, "Error: double allocate Eigen tensor");
    tensor = new ETensor6<int64_t>(shape[0], shape[1], shape[2], shape[3], shape[4], shape[5]);
    if (tensor)
        return 0;
    else
        return 1;
}

template <>
int TosaReference::Tensor0<bool>::allocate()
{
    ASSERT_MSG(tensor == nullptr, "Error: double allocate Eigen tensor");
    tensor = new ETensor0<bool>();
    if (tensor)
        return 0;
    else
        return 1;
}
template <>
int TosaReference::Tensor1<bool>::allocate()
{
    ASSERT_MSG(tensor == nullptr, "Error: double allocate Eigen tensor");
    tensor = new ETensor1<bool>(shape[0]);
    if (tensor)
        return 0;
    else
        return 1;
}
template <>
int TosaReference::Tensor2<bool>::allocate()
{
    ASSERT_MSG(tensor == nullptr, "Error: double allocate Eigen tensor");
    tensor = new ETensor2<bool>(shape[0], shape[1]);
    if (tensor)
        return 0;
    else
        return 1;
}

template <>
int TosaReference::Tensor3<bool>::allocate()
{
    ASSERT_MSG(tensor == nullptr, "Error: double allocate Eigen tensor");
    tensor = new ETensor3<bool>(shape[0], shape[1], shape[2]);
    if (tensor)
        return 0;
    else
        return 1;
}

template <>
int TosaReference::Tensor4<bool>::allocate()
{
    ASSERT_MSG(tensor == nullptr, "Error: double allocate Eigen tensor");
    tensor = new ETensor4<bool>(shape[0], shape[1], shape[2], shape[3]);
    if (tensor)
        return 0;
    else
        return 1;
}

template <>
int TosaReference::Tensor5<bool>::allocate()
{
    ASSERT_MSG(tensor == nullptr, "Error: double allocate Eigen tensor");
    tensor = new ETensor5<bool>(shape[0], shape[1], shape[2], shape[3], shape[4]);
    if (tensor)
        return 0;
    else
        return 1;
}

template <>
int TosaReference::Tensor6<bool>::allocate()
{
    ASSERT_MSG(tensor == nullptr, "Error: double allocate Eigen tensor");
    tensor = new ETensor6<bool>(shape[0], shape[1], shape[2], shape[3], shape[4], shape[5]);
    if (tensor)
        return 0;
    else
        return 1;
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

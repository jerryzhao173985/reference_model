// Copyright (c) 2020-2025, ARM Limited.
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

#ifndef TOSA_REFERENCE_TENSOR_H
#define TOSA_REFERENCE_TENSOR_H

#include "array_proxy.h"
#include "cfloat.h"
#include "config.h"
#include "dtype.h"
#include "model_common.h"
#include "ops/template_types.h"
#include "tosa_serialization_handler.h"
#include <list>
#include <vector>

#include <unsupported/Eigen/CXX11/Tensor>

using bf16    = ct::cfloat<int16_t, 8, true, true, true>;
using fp8e4m3 = ct::cfloat<int8_t, 4, true, true, false>;
using fp8e5m2 = ct::cfloat<int8_t, 5, true, true, true>;

using namespace tosa;

namespace TosaReference
{
class GraphNode;

class Tensor
{
public:
    Tensor(const std::string tensorName_, const DType serializationDtype_, const std::vector<int> shape_);

    virtual ~Tensor();

    int setIsSubgraphInput();
    int setIsSubgraphOutput();
    int setIsParentGraphOutput();

    bool getIsParentGraphOutput() const
    {
        return isParentGraphOutput;
    }
    int setIsVariable();

    bool getIsSubgraphInput() const
    {
        return isSubgraphInput;
    }

    bool getIsSubgraphOutput() const
    {
        return isSubgraphOutput;
    }

    bool getIsVariable() const
    {
        return isVariable;
    }

    int setProducer(GraphNode* node);
    int addConsumer(GraphNode* node);

    int setIsValid()
    {
        isValid = 1;
        return 0;
    }

    int clearIsValid()
    {
        isValid = 0;
        return 0;
    }

    int getIsValid() const
    {
        return isValid;
    }

    GraphNode* getProducer()
    {
        return producer;
    }

    std::vector<GraphNode*>& getConsumers()
    {
        return consumers;
    }

    const std::string& getName() const
    {
        return tensorName;
    }

    const std::vector<int>& getShape() const
    {
        return shape;
    }

    uint32_t getDimSize(size_t dim) const
    {
        assert(dim < this->shape.size() && "Invalid dimension to getDimSize()");
        return static_cast<uint32_t>(this->shape[dim]);
    }

    void setDimSize(size_t dim, uint32_t new_size)
    {
        this->shape[dim] = static_cast<int>(new_size);
        return;
    }

    void setShapeValue(std::vector<int>& shapeValue)
    {
        for (auto dim : shapeValue)
        {
            this->shapeValue.push_back(dim);
        }
        return;
    }

    int getShapeValueSize() const
    {
        return static_cast<int>(this->shapeValue.size());
    }

    std::string getShapeValueAsString() const
    {
        std::string shape_str("[");
        for (auto& dim : shapeValue)
        {
            shape_str += (std::to_string(dim) + ", ");
        }
        shape_str.append("]");
        return shape_str;
    }

    std::string getShapeAsString() const
    {
        std::string shape_str("[");
        for (auto& dim : shape)
        {
            shape_str += (std::to_string(dim) + ", ");
        }
        shape_str.append("]");
        return shape_str;
    }

    const uint32_t getElementCount() const
    {
        int32_t elements = 1;
        for (size_t i = 0; i < shape.size(); i++)
            elements *= shape[i];

        return static_cast<uint32_t>(elements);
    }

    // Comparison of rank and type with other tensors
    const int matchRank(const Tensor& ref) const
    {
        return (ref.shape.size() == shape.size()) ? 0 : 1;
    }

    const int matchType(const Tensor& ref) const
    {
        return (ref.tensorDtype == tensorDtype) ? 0 : 1;
    }

    const int matchRankType(const Tensor& ref) const
    {
        return (matchType(ref) || matchRank(ref));
    }

    const int matchRankTypeShape(const Tensor& ref, const bool broadcastOk = false) const
    {
        if (matchRankType(ref))
            return 1;

        for (size_t i = 0; i < shape.size(); i++)
        {
            if (shape[i] != ref.shape[i])
            {
                if (!broadcastOk ||
                    // For broadcasts, the order of *this and ref matters.
                    // *this should be the source tensor.
                    // ref should be the target tensor. In most of the case, ref is expected to be the output tensor.
                    // this->shape must have size 1 if they don't match
                    (broadcastOk && (shape[i] != 1)))
                {
                    return 1;
                }
            }
        }

        return 0;
    }

    const int matchRankShape(const Tensor& ref, const bool broadcastOk = false) const
    {
        if (matchRank(ref))
            return 1;

        for (size_t i = 0; i < shape.size(); i++)
        {
            if (shape[i] != ref.shape[i])
            {
                if (!broadcastOk ||
                    // For broadcasts, the order of *this and ref matters.
                    // *this should be the source tensor.
                    // ref should be the target tensor. In most of the case, ref is expected to be the output tensor.
                    // this->shape must have size 1 if they don't match
                    (broadcastOk && (shape[i] != 1)))
                {
                    return 1;
                }
            }
        }

        return 0;
    }

    // Sometimes we might want to match several semi-compatible types,
    // so just check rank and size here
    const int matchRankSize(const Tensor& ref) const
    {
        if (matchRank(ref))
            return 1;

        for (size_t i = 0; i < shape.size(); i++)
        {
            if (shape[i] != ref.shape[i])
                return 1;
        }

        return 0;
    }

    // Unary check to make sure rank matches
    const int checkRequiredRank(const int minRank) const
    {
        return (shape.size() >= (size_t)minRank) ? 0 : 1;
    }

    const int checkRequiredRank(const int minRank, const int maxRank) const
    {
        return (shape.size() >= (size_t)minRank && shape.size() <= (size_t)maxRank) ? 0 : 1;
    }

    const int getRank() const
    {
        return static_cast<int>(shape.size());
    }

    const TOSA_REF_TYPE getDtype() const
    {
        return tensorDtype;
    }

    const DType getSerializationDtype() const
    {
        return serializationDtype;
    }

    virtual int dumpTensor(FILE* out) const = 0;
    virtual int dumpTensorParams(FILE* out) const;
    virtual int dumpTensorParams(std::ostream& out) const;

    virtual int setTensorValueDouble(const size_t bufLen, const double* vals)   = 0;
    virtual int setTensorValueFloat(const size_t bufLen, const float* vals)     = 0;
    virtual int setTensorValueUInt8(const size_t bufLen, const uint8_t* vals)   = 0;
    virtual int setTensorValueInt8(const size_t bufLen, const int8_t* vals)     = 0;
    virtual int setTensorValueUInt16(const size_t bufLen, const uint16_t* vals) = 0;
    virtual int setTensorValueInt16(const size_t bufLen, const int16_t* vals)   = 0;
    virtual int setTensorValueInt32(const size_t bufLen, const int32_t* vals)   = 0;
    virtual int setTensorValueInt64(const size_t bufLen, const int64_t* vals)   = 0;
    virtual int setTensorValueBool(const size_t bufLen, const bool* vals)       = 0;
    virtual int getTensorValueDouble(const size_t bufLen, double* fbuf) const   = 0;
    virtual int getTensorValueFloat(const size_t bufLen, float* fbuf) const     = 0;
    virtual int getTensorValueUInt8(const size_t bufLen, uint8_t* ibuf) const   = 0;
    virtual int getTensorValueInt8(const size_t bufLen, int8_t* ibuf) const     = 0;
    virtual int getTensorValueUInt16(const size_t bufLen, uint16_t* ibuf) const = 0;
    virtual int getTensorValueInt16(const size_t bufLen, int16_t* ibuf) const   = 0;
    virtual int getTensorValueInt32(const size_t bufLen, int32_t* ibuf) const   = 0;
    virtual int getTensorValueInt64(const size_t bufLen, int64_t* ibuf) const   = 0;
    virtual int getTensorValueBool(const size_t bufLen, bool* ibuf) const       = 0;

    virtual int readFromNpyFile(const char* filename);
    virtual int writeToNpyFile(const char* filename) const;
    virtual int copyValueFrom(Tensor* tensor) = 0;

    virtual int readfromVector(const ArrayProxy<double> vals);
    virtual int readfromVector(const ArrayProxy<float> vals);
    virtual int readfromVector(const ArrayProxy<half_float::half> vals);
    virtual int readfromVector(const ArrayProxy<int8_t> vals);
    virtual int readfromVector(const ArrayProxy<uint16_t> vals);
    virtual int readfromVector(const ArrayProxy<int16_t> vals);
    virtual int readfromVector(const ArrayProxy<int32_t> vals);
    virtual int readfromVector(const ArrayProxy<int64_t> vals);
    virtual int readfromVector(const ArrayProxy<unsigned char> vals);
    virtual int readfromVector(const ArrayProxy<bf16> vals);
    virtual int readfromVector(const ArrayProxy<fp8e4m3> vals);
    virtual int readfromVector(const ArrayProxy<fp8e5m2> vals);

    virtual int writeToVector(ArrayProxy<double> vals);
    virtual int writeToVector(ArrayProxy<float> vals);
    virtual int writeToVector(ArrayProxy<half_float::half> vals);
    virtual int writeToVector(ArrayProxy<int8_t> vals);
    virtual int writeToVector(ArrayProxy<uint16_t> vals);
    virtual int writeToVector(ArrayProxy<int16_t> vals);
    virtual int writeToVector(ArrayProxy<int32_t> vals);
    virtual int writeToVector(ArrayProxy<int64_t> vals);
    virtual int writeToVector(ArrayProxy<unsigned char> vals);
    virtual int writeToVector(ArrayProxy<bf16> vals);
    virtual int writeToVector(ArrayProxy<fp8e4m3> vals);
    virtual int writeToVector(ArrayProxy<fp8e5m2> vals);

    const char* bool_to_str(bool in) const
    {
        static const char* true_str  = "true";
        static const char* false_str = "false";
        return in ? true_str : false_str;
    }

    virtual int allocate()            = 0;
    virtual int deallocate()          = 0;
    virtual bool is_allocated() const = 0;

protected:
    const std::string tensorName;
    const DType serializationDtype;
    std::vector<int> shape;
    std::vector<int> shapeValue;
    const TOSA_REF_TYPE tensorDtype;
    bool isValid;
    bool isSubgraphInput;
    bool isSubgraphOutput;
    bool isVariable;
    bool isAllocated;

    bool isParentGraphOutput;

    GraphNode* producer;
    std::vector<GraphNode*> consumers;

    // Note: the Eigen::Tensor is not declared in Tensor
    // Instead, the TensorTemplate class keeps the templated tensor
    // declaration so that the graph manipulation tools are isolated
    // from the templated tensor type.
    //
    // Operators need to be aware of the TensorTemplate<EigenTensor<type, rank>> type
    // so that they can operate on the right types.
};

template <class T>
class TensorTemplate : public Tensor
{
public:
    TensorTemplate(const std::string tensorName_, const DType dtype_, const std::vector<int> shape_)
        : Tensor(tensorName_, dtype_, shape_)
    {
        tensor = nullptr;
    }

    virtual ~TensorTemplate()
    {
        deallocate();
    }

    virtual int allocate()
    {
        tensor = new T();
        if (tensor)
            return 0;
        else
            return 1;
    }

    virtual int deallocate()
    {
        if (tensor)
        {
            DEBUG_INFO(GT, "Deallocating tensor %s", tensorName.c_str());
            delete tensor;
        }
        tensor = nullptr;
        return 0;
    }

    virtual bool is_allocated() const
    {
        if (tensor)
        {
            return true;
        }
        return false;
    }

    T& getTensor()
    {
        return *tensor;
    }

    virtual int dumpTensor(FILE* out) const;

    virtual int setTensorValueDouble(const size_t bufLen, const double* vals);
    virtual int setTensorValueFloat(const size_t bufLen, const float* vals);
    virtual int setTensorValueUInt8(const size_t bufLen, const uint8_t* vals);
    virtual int setTensorValueInt8(const size_t bufLen, const int8_t* vals);
    virtual int setTensorValueUInt16(const size_t bufLen, const uint16_t* vals);
    virtual int setTensorValueInt16(const size_t bufLen, const int16_t* vals);
    virtual int setTensorValueInt32(const size_t bufLen, const int32_t* vals);
    virtual int setTensorValueInt64(const size_t bufLen, const int64_t* vals);
    virtual int setTensorValueBool(const size_t bufLen, const bool* vals);

    virtual int getTensorValueDouble(const size_t bufLen, double* fbuf) const;
    virtual int getTensorValueFloat(const size_t bufLen, float* fbuf) const;
    virtual int getTensorValueUInt8(const size_t bufLen, uint8_t* ibuf) const;
    virtual int getTensorValueInt8(const size_t bufLen, int8_t* ibuf) const;
    virtual int getTensorValueUInt16(const size_t bufLen, uint16_t* ibuf) const;
    virtual int getTensorValueInt16(const size_t bufLen, int16_t* ibuf) const;
    virtual int getTensorValueInt32(const size_t bufLen, int32_t* ibuf) const;
    virtual int getTensorValueInt64(const size_t bufLen, int64_t* ibuf) const;
    virtual int getTensorValueBool(const size_t bufLen, bool* bbuf) const;

    virtual int copyValueFrom(Tensor* tensor);

protected:
    T* tensor;
};

// allocate() template specializations to allocate the different tensor sizes
// Let the compiler know here before the factory uses them, but define them in the .cc file.
template <>
int Tensor0<float>::allocate();
template <>
int Tensor1<float>::allocate();
template <>
int Tensor2<float>::allocate();
template <>
int Tensor3<float>::allocate();
template <>
int Tensor4<float>::allocate();
template <>
int Tensor5<float>::allocate();
template <>
int Tensor6<float>::allocate();

template <>
int Tensor0<double>::allocate();
template <>
int Tensor1<double>::allocate();
template <>
int Tensor2<double>::allocate();
template <>
int Tensor3<double>::allocate();
template <>
int Tensor4<double>::allocate();
template <>
int Tensor5<double>::allocate();
template <>
int Tensor6<double>::allocate();

template <>
int Tensor0<int32_t>::allocate();
template <>
int Tensor1<int32_t>::allocate();
template <>
int Tensor2<int32_t>::allocate();
template <>
int Tensor3<int32_t>::allocate();
template <>
int Tensor4<int32_t>::allocate();
template <>
int Tensor5<int32_t>::allocate();
template <>
int Tensor6<int32_t>::allocate();

template <>
int Tensor0<int64_t>::allocate();
template <>
int Tensor1<int64_t>::allocate();
template <>
int Tensor2<int64_t>::allocate();
template <>
int Tensor3<int64_t>::allocate();
template <>
int Tensor4<int64_t>::allocate();
template <>
int Tensor5<int64_t>::allocate();
template <>
int Tensor6<int64_t>::allocate();

template <>
int Tensor0<bool>::allocate();
template <>
int Tensor1<bool>::allocate();
template <>
int Tensor2<bool>::allocate();
template <>
int Tensor3<bool>::allocate();
template <>
int Tensor4<bool>::allocate();
template <>
int Tensor5<bool>::allocate();
template <>
int Tensor6<bool>::allocate();

template <>
int Tensor0<float>::copyValueFrom(Tensor* src);
template <>
int Tensor1<float>::copyValueFrom(Tensor* src);
template <>
int Tensor2<float>::copyValueFrom(Tensor* src);
template <>
int Tensor3<float>::copyValueFrom(Tensor* src);
template <>
int Tensor4<float>::copyValueFrom(Tensor* src);
template <>
int Tensor5<float>::copyValueFrom(Tensor* src);
template <>
int Tensor6<float>::copyValueFrom(Tensor* src);

template <>
int Tensor0<double>::copyValueFrom(Tensor* src);
template <>
int Tensor1<double>::copyValueFrom(Tensor* src);
template <>
int Tensor2<double>::copyValueFrom(Tensor* src);
template <>
int Tensor3<double>::copyValueFrom(Tensor* src);
template <>
int Tensor4<double>::copyValueFrom(Tensor* src);
template <>
int Tensor5<double>::copyValueFrom(Tensor* src);
template <>
int Tensor6<double>::copyValueFrom(Tensor* src);

template <>
int Tensor0<int32_t>::copyValueFrom(Tensor* src);
template <>
int Tensor1<int32_t>::copyValueFrom(Tensor* src);
template <>
int Tensor2<int32_t>::copyValueFrom(Tensor* src);
template <>
int Tensor3<int32_t>::copyValueFrom(Tensor* src);
template <>
int Tensor4<int32_t>::copyValueFrom(Tensor* src);
template <>
int Tensor5<int32_t>::copyValueFrom(Tensor* src);
template <>
int Tensor6<int32_t>::copyValueFrom(Tensor* src);

template <>
int Tensor0<int64_t>::copyValueFrom(Tensor* src);
template <>
int Tensor1<int64_t>::copyValueFrom(Tensor* src);
template <>
int Tensor2<int64_t>::copyValueFrom(Tensor* src);
template <>
int Tensor3<int64_t>::copyValueFrom(Tensor* src);
template <>
int Tensor4<int64_t>::copyValueFrom(Tensor* src);
template <>
int Tensor5<int64_t>::copyValueFrom(Tensor* src);
template <>
int Tensor6<int64_t>::copyValueFrom(Tensor* src);

template <>
int Tensor0<bool>::copyValueFrom(Tensor* src);
template <>
int Tensor1<bool>::copyValueFrom(Tensor* src);
template <>
int Tensor2<bool>::copyValueFrom(Tensor* src);
template <>
int Tensor3<bool>::copyValueFrom(Tensor* src);
template <>
int Tensor4<bool>::copyValueFrom(Tensor* src);
template <>
int Tensor5<bool>::copyValueFrom(Tensor* src);
template <>
int Tensor6<bool>::copyValueFrom(Tensor* src);

template <>
int Tensor0<int32_t>::setTensorValueUInt8(const size_t bufLen, const uint8_t* vals);
template <>
int Tensor1<int32_t>::setTensorValueUInt8(const size_t bufLen, const uint8_t* vals);
template <>
int Tensor2<int32_t>::setTensorValueUInt8(const size_t bufLen, const uint8_t* vals);
template <>
int Tensor3<int32_t>::setTensorValueUInt8(const size_t bufLen, const uint8_t* vals);
template <>
int Tensor4<int32_t>::setTensorValueUInt8(const size_t bufLen, const uint8_t* vals);
template <>
int Tensor5<int32_t>::setTensorValueUInt8(const size_t bufLen, const uint8_t* vals);
template <>
int Tensor6<int32_t>::setTensorValueUInt8(const size_t bufLen, const uint8_t* vals);

template <>
int Tensor0<int32_t>::setTensorValueInt8(const size_t bufLen, const int8_t* vals);
template <>
int Tensor1<int32_t>::setTensorValueInt8(const size_t bufLen, const int8_t* vals);
template <>
int Tensor2<int32_t>::setTensorValueInt8(const size_t bufLen, const int8_t* vals);
template <>
int Tensor3<int32_t>::setTensorValueInt8(const size_t bufLen, const int8_t* vals);
template <>
int Tensor4<int32_t>::setTensorValueInt8(const size_t bufLen, const int8_t* vals);
template <>
int Tensor5<int32_t>::setTensorValueInt8(const size_t bufLen, const int8_t* vals);
template <>
int Tensor6<int32_t>::setTensorValueInt8(const size_t bufLen, const int8_t* vals);

template <>
int Tensor0<int32_t>::setTensorValueUInt16(const size_t bufLen, const uint16_t* vals);
template <>
int Tensor1<int32_t>::setTensorValueUInt16(const size_t bufLen, const uint16_t* vals);
template <>
int Tensor2<int32_t>::setTensorValueUInt16(const size_t bufLen, const uint16_t* vals);
template <>
int Tensor3<int32_t>::setTensorValueUInt16(const size_t bufLen, const uint16_t* vals);
template <>
int Tensor4<int32_t>::setTensorValueUInt16(const size_t bufLen, const uint16_t* vals);
template <>
int Tensor5<int32_t>::setTensorValueUInt16(const size_t bufLen, const uint16_t* vals);
template <>
int Tensor6<int32_t>::setTensorValueUInt16(const size_t bufLen, const uint16_t* vals);

template <>
int Tensor0<int32_t>::setTensorValueInt16(const size_t bufLen, const int16_t* vals);
template <>
int Tensor1<int32_t>::setTensorValueInt16(const size_t bufLen, const int16_t* vals);
template <>
int Tensor2<int32_t>::setTensorValueInt16(const size_t bufLen, const int16_t* vals);
template <>
int Tensor3<int32_t>::setTensorValueInt16(const size_t bufLen, const int16_t* vals);
template <>
int Tensor4<int32_t>::setTensorValueInt16(const size_t bufLen, const int16_t* vals);
template <>
int Tensor5<int32_t>::setTensorValueInt16(const size_t bufLen, const int16_t* vals);
template <>
int Tensor6<int32_t>::setTensorValueInt16(const size_t bufLen, const int16_t* vals);

template <>
int Tensor0<int32_t>::setTensorValueInt32(const size_t bufLen, const int32_t* vals);
template <>
int Tensor1<int32_t>::setTensorValueInt32(const size_t bufLen, const int32_t* vals);
template <>
int Tensor2<int32_t>::setTensorValueInt32(const size_t bufLen, const int32_t* vals);
template <>
int Tensor3<int32_t>::setTensorValueInt32(const size_t bufLen, const int32_t* vals);
template <>
int Tensor4<int32_t>::setTensorValueInt32(const size_t bufLen, const int32_t* vals);
template <>
int Tensor5<int32_t>::setTensorValueInt32(const size_t bufLen, const int32_t* vals);
template <>
int Tensor6<int32_t>::setTensorValueInt32(const size_t bufLen, const int32_t* vals);

template <>
int Tensor0<int32_t>::getTensorValueUInt8(const size_t bufLen, uint8_t* vals) const;
template <>
int Tensor1<int32_t>::getTensorValueUInt8(const size_t bufLen, uint8_t* vals) const;
template <>
int Tensor2<int32_t>::getTensorValueUInt8(const size_t bufLen, uint8_t* vals) const;
template <>
int Tensor3<int32_t>::getTensorValueUInt8(const size_t bufLen, uint8_t* vals) const;
template <>
int Tensor4<int32_t>::getTensorValueUInt8(const size_t bufLen, uint8_t* vals) const;
template <>
int Tensor5<int32_t>::getTensorValueUInt8(const size_t bufLen, uint8_t* vals) const;
template <>
int Tensor6<int32_t>::getTensorValueUInt8(const size_t bufLen, uint8_t* vals) const;

template <>
int Tensor0<int32_t>::getTensorValueInt8(const size_t bufLen, int8_t* vals) const;
template <>
int Tensor1<int32_t>::getTensorValueInt8(const size_t bufLen, int8_t* vals) const;
template <>
int Tensor2<int32_t>::getTensorValueInt8(const size_t bufLen, int8_t* vals) const;
template <>
int Tensor3<int32_t>::getTensorValueInt8(const size_t bufLen, int8_t* vals) const;
template <>
int Tensor4<int32_t>::getTensorValueInt8(const size_t bufLen, int8_t* vals) const;
template <>
int Tensor5<int32_t>::getTensorValueInt8(const size_t bufLen, int8_t* vals) const;
template <>
int Tensor6<int32_t>::getTensorValueInt8(const size_t bufLen, int8_t* vals) const;

template <>
int Tensor0<int32_t>::getTensorValueUInt16(const size_t bufLen, uint16_t* vals) const;
template <>
int Tensor1<int32_t>::getTensorValueUInt16(const size_t bufLen, uint16_t* vals) const;
template <>
int Tensor2<int32_t>::getTensorValueUInt16(const size_t bufLen, uint16_t* vals) const;
template <>
int Tensor3<int32_t>::getTensorValueUInt16(const size_t bufLen, uint16_t* vals) const;
template <>
int Tensor4<int32_t>::getTensorValueUInt16(const size_t bufLen, uint16_t* vals) const;
template <>
int Tensor5<int32_t>::getTensorValueUInt16(const size_t bufLen, uint16_t* vals) const;
template <>
int Tensor6<int32_t>::getTensorValueUInt16(const size_t bufLen, uint16_t* vals) const;

template <>
int Tensor0<int32_t>::getTensorValueInt16(const size_t bufLen, int16_t* vals) const;
template <>
int Tensor1<int32_t>::getTensorValueInt16(const size_t bufLen, int16_t* vals) const;
template <>
int Tensor2<int32_t>::getTensorValueInt16(const size_t bufLen, int16_t* vals) const;
template <>
int Tensor3<int32_t>::getTensorValueInt16(const size_t bufLen, int16_t* vals) const;
template <>
int Tensor4<int32_t>::getTensorValueInt16(const size_t bufLen, int16_t* vals) const;
template <>
int Tensor5<int32_t>::getTensorValueInt16(const size_t bufLen, int16_t* vals) const;
template <>
int Tensor6<int32_t>::getTensorValueInt16(const size_t bufLen, int16_t* vals) const;

template <>
int Tensor0<int32_t>::getTensorValueInt32(const size_t bufLen, int32_t* vals) const;
template <>
int Tensor1<int32_t>::getTensorValueInt32(const size_t bufLen, int32_t* vals) const;
template <>
int Tensor2<int32_t>::getTensorValueInt32(const size_t bufLen, int32_t* vals) const;
template <>
int Tensor3<int32_t>::getTensorValueInt32(const size_t bufLen, int32_t* vals) const;
template <>
int Tensor4<int32_t>::getTensorValueInt32(const size_t bufLen, int32_t* vals) const;
template <>
int Tensor5<int32_t>::getTensorValueInt32(const size_t bufLen, int32_t* vals) const;
template <>
int Tensor6<int32_t>::getTensorValueInt32(const size_t bufLen, int32_t* vals) const;

template <>
int Tensor0<int64_t>::setTensorValueInt64(const size_t bufLen, const int64_t* vals);
template <>
int Tensor1<int64_t>::setTensorValueInt64(const size_t bufLen, const int64_t* vals);
template <>
int Tensor2<int64_t>::setTensorValueInt64(const size_t bufLen, const int64_t* vals);
template <>
int Tensor3<int64_t>::setTensorValueInt64(const size_t bufLen, const int64_t* vals);
template <>
int Tensor4<int64_t>::setTensorValueInt64(const size_t bufLen, const int64_t* vals);
template <>
int Tensor5<int64_t>::setTensorValueInt64(const size_t bufLen, const int64_t* vals);
template <>
int Tensor6<int64_t>::setTensorValueInt64(const size_t bufLen, const int64_t* vals);

template <>
int Tensor0<int64_t>::getTensorValueInt64(const size_t bufLen, int64_t* vals) const;
template <>
int Tensor1<int64_t>::getTensorValueInt64(const size_t bufLen, int64_t* vals) const;
template <>
int Tensor2<int64_t>::getTensorValueInt64(const size_t bufLen, int64_t* vals) const;
template <>
int Tensor3<int64_t>::getTensorValueInt64(const size_t bufLen, int64_t* vals) const;
template <>
int Tensor4<int64_t>::getTensorValueInt64(const size_t bufLen, int64_t* vals) const;
template <>
int Tensor5<int64_t>::getTensorValueInt64(const size_t bufLen, int64_t* vals) const;
template <>
int Tensor6<int64_t>::getTensorValueInt64(const size_t bufLen, int64_t* vals) const;

template <>
int Tensor0<float>::setTensorValueFloat(const size_t bufLen, const float* vals);
template <>
int Tensor1<float>::setTensorValueFloat(const size_t bufLen, const float* vals);
template <>
int Tensor2<float>::setTensorValueFloat(const size_t bufLen, const float* vals);
template <>
int Tensor3<float>::setTensorValueFloat(const size_t bufLen, const float* vals);
template <>
int Tensor4<float>::setTensorValueFloat(const size_t bufLen, const float* vals);
template <>
int Tensor5<float>::setTensorValueFloat(const size_t bufLen, const float* vals);
template <>
int Tensor6<float>::setTensorValueFloat(const size_t bufLen, const float* vals);

template <>
int Tensor0<float>::getTensorValueFloat(const size_t bufLen, float* vals) const;
template <>
int Tensor1<float>::getTensorValueFloat(const size_t bufLen, float* vals) const;
template <>
int Tensor2<float>::getTensorValueFloat(const size_t bufLen, float* vals) const;
template <>
int Tensor3<float>::getTensorValueFloat(const size_t bufLen, float* vals) const;
template <>
int Tensor4<float>::getTensorValueFloat(const size_t bufLen, float* vals) const;
template <>
int Tensor5<float>::getTensorValueFloat(const size_t bufLen, float* vals) const;
template <>
int Tensor6<float>::getTensorValueFloat(const size_t bufLen, float* vals) const;

template <>
int Tensor0<double>::setTensorValueDouble(const size_t bufLen, const double* vals);
template <>
int Tensor1<double>::setTensorValueDouble(const size_t bufLen, const double* vals);
template <>
int Tensor2<double>::setTensorValueDouble(const size_t bufLen, const double* vals);
template <>
int Tensor3<double>::setTensorValueDouble(const size_t bufLen, const double* vals);
template <>
int Tensor4<double>::setTensorValueDouble(const size_t bufLen, const double* vals);
template <>
int Tensor5<double>::setTensorValueDouble(const size_t bufLen, const double* vals);
template <>
int Tensor6<double>::setTensorValueDouble(const size_t bufLen, const double* vals);

template <>
int Tensor0<double>::getTensorValueDouble(const size_t bufLen, double* vals) const;
template <>
int Tensor1<double>::getTensorValueDouble(const size_t bufLen, double* vals) const;
template <>
int Tensor2<double>::getTensorValueDouble(const size_t bufLen, double* vals) const;
template <>
int Tensor3<double>::getTensorValueDouble(const size_t bufLen, double* vals) const;
template <>
int Tensor4<double>::getTensorValueDouble(const size_t bufLen, double* vals) const;
template <>
int Tensor5<double>::getTensorValueDouble(const size_t bufLen, double* vals) const;
template <>
int Tensor6<double>::getTensorValueDouble(const size_t bufLen, double* vals) const;

template <>
int Tensor0<bool>::setTensorValueBool(const size_t bufLen, const bool* vals);
template <>
int Tensor1<bool>::setTensorValueBool(const size_t bufLen, const bool* vals);
template <>
int Tensor2<bool>::setTensorValueBool(const size_t bufLen, const bool* vals);
template <>
int Tensor3<bool>::setTensorValueBool(const size_t bufLen, const bool* vals);
template <>
int Tensor4<bool>::setTensorValueBool(const size_t bufLen, const bool* vals);
template <>
int Tensor5<bool>::setTensorValueBool(const size_t bufLen, const bool* vals);
template <>
int Tensor6<bool>::setTensorValueBool(const size_t bufLen, const bool* vals);

template <>
int Tensor0<bool>::getTensorValueBool(const size_t bufLen, bool* vals) const;
template <>
int Tensor1<bool>::getTensorValueBool(const size_t bufLen, bool* vals) const;
template <>
int Tensor2<bool>::getTensorValueBool(const size_t bufLen, bool* vals) const;
template <>
int Tensor3<bool>::getTensorValueBool(const size_t bufLen, bool* vals) const;
template <>
int Tensor4<bool>::getTensorValueBool(const size_t bufLen, bool* vals) const;
template <>
int Tensor5<bool>::getTensorValueBool(const size_t bufLen, bool* vals) const;
template <>
int Tensor6<bool>::getTensorValueBool(const size_t bufLen, bool* vals) const;

template <>
int Tensor0<float>::dumpTensor(FILE* out) const;
template <>
int Tensor1<float>::dumpTensor(FILE* out) const;
template <>
int Tensor2<float>::dumpTensor(FILE* out) const;
template <>
int Tensor3<float>::dumpTensor(FILE* out) const;
template <>
int Tensor4<float>::dumpTensor(FILE* out) const;
template <>
int Tensor5<float>::dumpTensor(FILE* out) const;
template <>
int Tensor6<float>::dumpTensor(FILE* out) const;
template <>
int Tensor0<double>::dumpTensor(FILE* out) const;
template <>
int Tensor1<double>::dumpTensor(FILE* out) const;
template <>
int Tensor2<double>::dumpTensor(FILE* out) const;
template <>
int Tensor3<double>::dumpTensor(FILE* out) const;
template <>
int Tensor4<double>::dumpTensor(FILE* out) const;
template <>
int Tensor5<float>::dumpTensor(FILE* out) const;
template <>
int Tensor6<double>::dumpTensor(FILE* out) const;
template <>
int Tensor0<int32_t>::dumpTensor(FILE* out) const;
template <>
int Tensor1<int32_t>::dumpTensor(FILE* out) const;
template <>
int Tensor2<int32_t>::dumpTensor(FILE* out) const;
template <>
int Tensor3<int32_t>::dumpTensor(FILE* out) const;
template <>
int Tensor4<int32_t>::dumpTensor(FILE* out) const;
template <>
int Tensor5<int32_t>::dumpTensor(FILE* out) const;
template <>
int Tensor6<int32_t>::dumpTensor(FILE* out) const;
template <>
int Tensor0<int64_t>::dumpTensor(FILE* out) const;
template <>
int Tensor1<int64_t>::dumpTensor(FILE* out) const;
template <>
int Tensor2<int64_t>::dumpTensor(FILE* out) const;
template <>
int Tensor3<int64_t>::dumpTensor(FILE* out) const;
template <>
int Tensor4<int64_t>::dumpTensor(FILE* out) const;
template <>
int Tensor5<int64_t>::dumpTensor(FILE* out) const;
template <>
int Tensor6<int64_t>::dumpTensor(FILE* out) const;
template <>
int Tensor0<bool>::dumpTensor(FILE* out) const;
template <>
int Tensor1<bool>::dumpTensor(FILE* out) const;
template <>
int Tensor2<bool>::dumpTensor(FILE* out) const;
template <>
int Tensor3<bool>::dumpTensor(FILE* out) const;
template <>
int Tensor4<bool>::dumpTensor(FILE* out) const;
template <>
int Tensor5<bool>::dumpTensor(FILE* out) const;
template <>
int Tensor6<bool>::dumpTensor(FILE* out) const;

class TensorFactory
{
public:
    static Tensor* newTensor(std::string tensorName_, DType dtype_, std::vector<int> shape_, const uint32_t rank)
    {
        TOSA_REF_TYPE tensorDtype_ = ConvertDType(dtype_);
        switch (tensorDtype_)
        {
            case TOSA_REF_TYPE_FP32:
            case TOSA_REF_TYPE_FP16:
            case TOSA_REF_TYPE_BF16:
            case TOSA_REF_TYPE_FP8E4M3:
            case TOSA_REF_TYPE_FP8E5M2:
                switch (rank)
                {
                    case 0:
                        return new Tensor0<float>(tensorName_, dtype_, shape_);
                    case 1:
                        return new Tensor1<float>(tensorName_, dtype_, shape_);
                    case 2:
                        return new Tensor2<float>(tensorName_, dtype_, shape_);
                    case 3:
                        return new Tensor3<float>(tensorName_, dtype_, shape_);
                    case 4:
                        return new Tensor4<float>(tensorName_, dtype_, shape_);
                    case 5:
                        return new Tensor5<float>(tensorName_, dtype_, shape_);
                    case 6:
                        return new Tensor6<float>(tensorName_, dtype_, shape_);
                }
                break;
            case TOSA_REF_TYPE_INT32:
            case TOSA_REF_TYPE_UINT8:
            case TOSA_REF_TYPE_INT4:
            case TOSA_REF_TYPE_INT8:
            case TOSA_REF_TYPE_INT16:
            case TOSA_REF_TYPE_UINT16:
                switch (rank)
                {
                    case 0:
                        return new Tensor0<int32_t>(tensorName_, dtype_, shape_);
                    case 1:
                        return new Tensor1<int32_t>(tensorName_, dtype_, shape_);
                    case 2:
                        return new Tensor2<int32_t>(tensorName_, dtype_, shape_);
                    case 3:
                        return new Tensor3<int32_t>(tensorName_, dtype_, shape_);
                    case 4:
                        return new Tensor4<int32_t>(tensorName_, dtype_, shape_);
                    case 5:
                        return new Tensor5<int32_t>(tensorName_, dtype_, shape_);
                    case 6:
                        return new Tensor6<int32_t>(tensorName_, dtype_, shape_);
                }
                break;
            case TOSA_REF_TYPE_INT48:
                switch (rank)
                {
                    case 0:
                        return new Tensor0<int64_t>(tensorName_, dtype_, shape_);
                    case 1:
                        return new Tensor1<int64_t>(tensorName_, dtype_, shape_);
                    case 2:
                        return new Tensor2<int64_t>(tensorName_, dtype_, shape_);
                    case 3:
                        return new Tensor3<int64_t>(tensorName_, dtype_, shape_);
                    case 4:
                        return new Tensor4<int64_t>(tensorName_, dtype_, shape_);
                    case 5:
                        return new Tensor5<int64_t>(tensorName_, dtype_, shape_);
                    case 6:
                        return new Tensor6<int64_t>(tensorName_, dtype_, shape_);
                }
                break;
            case TOSA_REF_TYPE_SHAPE:
                switch (rank)
                {
                    case 0:
                        // <TOPIC: EMPTY_SHAPE>
                        // an empty shape (i.e. the shape of a rank 0 tensor) is
                        // itself encoded as {shape=[], data=[]}, which has the
                        // shape of a rank 0 tensor, but no data. To avoid having
                        // to deal with it as a special case, allocate it as an
                        // empty rank 1 here instead.
                        return new Tensor1<int64_t>(tensorName_, dtype_, { 0 });

                    case 1:
                        return new Tensor1<int64_t>(tensorName_, dtype_, shape_);
                    default:
                        break;
                }
                break;
            case TOSA_REF_TYPE_BOOL:
                switch (rank)
                {
                    case 0:
                        return new Tensor0<bool>(tensorName_, dtype_, shape_);
                    case 1:
                        return new Tensor1<bool>(tensorName_, dtype_, shape_);
                    case 2:
                        return new Tensor2<bool>(tensorName_, dtype_, shape_);
                    case 3:
                        return new Tensor3<bool>(tensorName_, dtype_, shape_);
                    case 4:
                        return new Tensor4<bool>(tensorName_, dtype_, shape_);
                    case 5:
                        return new Tensor5<bool>(tensorName_, dtype_, shape_);
                    case 6:
                        return new Tensor6<bool>(tensorName_, dtype_, shape_);
                }
                break;
            case TOSA_REF_TYPE_FP64:
                switch (rank)
                {
                    case 0:
                        return new Tensor0<double>(tensorName_, dtype_, shape_);
                    case 1:
                        return new Tensor1<double>(tensorName_, dtype_, shape_);
                    case 2:
                        return new Tensor2<double>(tensorName_, dtype_, shape_);
                    case 3:
                        return new Tensor3<double>(tensorName_, dtype_, shape_);
                    case 4:
                        return new Tensor4<double>(tensorName_, dtype_, shape_);
                    case 5:
                        return new Tensor5<double>(tensorName_, dtype_, shape_);
                    case 6:
                        return new Tensor6<double>(tensorName_, dtype_, shape_);
                }
                break;
            case TOSA_REF_TYPE_UNKNOWN:
                // tensorDtype_ is uninitialized
                break;
        }
        return nullptr;
    }
};
};    // namespace TosaReference

#endif

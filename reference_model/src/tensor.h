
// Copyright (c) 2020-2021, ARM Limited.
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

#include "model_common.h"
#include "ops/template_types.h"
#include "tosa_generated.h"
#include "tosa_serialization_handler.h"
#include <Eigen/CXX11/Tensor>
#include <list>
#include <vector>

using namespace tosa;

namespace TosaReference
{
class GraphNode;

class Tensor
{
public:
    Tensor(std::string tensorName_, DType tensorDtype__, std::vector<int> shape_);

    virtual ~Tensor();

    int setIsSubgraphInput();
    int setIsSubgraphOutput();

    int getIsSubgraphInput() const
    {
        return isSubgraphInput;
    }

    int getIsSubgraphOutput() const
    {
        return isSubgraphOutput;
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
        uint32_t elements = 1;
        for (size_t i = 0; i < shape.size(); i++)
            elements *= shape[i];

        return elements;
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
    const int checkRequiredRank(const int exactRank) const
    {
        return (shape.size() == (size_t)exactRank) ? 0 : 1;
    }

    const int checkRequiredRank(const int minRank, const int maxRank) const
    {
        return (shape.size() >= (size_t)minRank && shape.size() <= (size_t)maxRank) ? 0 : 1;
    }

    const int getRank() const
    {
        return shape.size();
    }

    const DType getDtype() const
    {
        return tensorDtype;
    }

    virtual int dumpTensor(FILE* out) const = 0;
    virtual int dumpTensorParams(FILE* out) const;
    virtual int dumpTensorParams(std::ostream& out) const;

    virtual int setTensorValueFloat(const size_t bufLen, const float* vals)   = 0;
    virtual int setTensorValueInt32(const size_t bufLen, const int32_t* vals) = 0;
    virtual int setTensorValueInt64(const size_t bufLen, const int64_t* vals) = 0;
    virtual int setTensorValueBool(const size_t bufLen, const bool* vals)     = 0;
    virtual int getTensorValueFloat(const size_t bufLen, float* fbuf) const   = 0;
    virtual int getTensorValueInt32(const size_t bufLen, int32_t* ibuf) const = 0;
    virtual int getTensorValueInt64(const size_t bufLen, int64_t* ibuf) const = 0;
    virtual int getTensorValueBool(const size_t bufLen, bool* ibuf) const     = 0;

    virtual int readFromNpyFile(const char* filename);
    virtual int writeToNpyFile(const char* filename) const;
    virtual int copyValueFrom(Tensor* tensor) = 0;

    const char* bool_to_str(bool in) const
    {
        static const char* true_str  = "true";
        static const char* false_str = "false";
        return in ? true_str : false_str;
    }

    virtual int allocate()      = 0;
    virtual int deallocate()    = 0;
    virtual bool is_allocated() = 0;

protected:
    std::string tensorName;
    DType tensorDtype;
    int isValid;
    std::vector<int> shape;
    int isSubgraphInput;
    int isSubgraphOutput;
    bool isAllocated;

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
    TensorTemplate(std::string tensorName_, DType tensorDtype_, std::vector<int> shape_)
        : Tensor(tensorName_, tensorDtype_, shape_)
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
            delete tensor;
        }
        tensor = nullptr;
        return 0;
    }

    virtual bool is_allocated()
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

    virtual int setTensorValueFloat(const size_t bufLen, const float* vals);
    virtual int setTensorValueInt32(const size_t bufLen, const int32_t* vals);
    virtual int setTensorValueInt64(const size_t bufLen, const int64_t* vals);
    virtual int setTensorValueBool(const size_t bufLen, const bool* vals);
    virtual int getTensorValueFloat(const size_t bufLen, float* fbuf) const;
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

// assume we only dump float type tensor now
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
    static Tensor* newTensor(std::string tensorName_, DType tensorDtype_, std::vector<int> shape_, const uint32_t rank)
    {
        switch (tensorDtype_)
        {
            case DType_FLOAT:
                switch (rank)
                {
                    case 0:
                        return new Tensor0<float>(tensorName_, tensorDtype_, shape_);
                    case 1:
                        return new Tensor1<float>(tensorName_, tensorDtype_, shape_);
                    case 2:
                        return new Tensor2<float>(tensorName_, tensorDtype_, shape_);
                    case 3:
                        return new Tensor3<float>(tensorName_, tensorDtype_, shape_);
                    case 4:
                        return new Tensor4<float>(tensorName_, tensorDtype_, shape_);
                    case 5:
                        return new Tensor5<float>(tensorName_, tensorDtype_, shape_);
                    case 6:
                        return new Tensor6<float>(tensorName_, tensorDtype_, shape_);
                }
                break;
            case DType_INT32:
            case DType_UINT8:
            case DType_INT4:
            case DType_INT8:
            case DType_INT16:
                switch (rank)
                {
                    case 0:
                        return new Tensor0<int32_t>(tensorName_, tensorDtype_, shape_);
                    case 1:
                        return new Tensor1<int32_t>(tensorName_, tensorDtype_, shape_);
                    case 2:
                        return new Tensor2<int32_t>(tensorName_, tensorDtype_, shape_);
                    case 3:
                        return new Tensor3<int32_t>(tensorName_, tensorDtype_, shape_);
                    case 4:
                        return new Tensor4<int32_t>(tensorName_, tensorDtype_, shape_);
                    case 5:
                        return new Tensor5<int32_t>(tensorName_, tensorDtype_, shape_);
                    case 6:
                        return new Tensor6<int32_t>(tensorName_, tensorDtype_, shape_);
                }
                break;
            case DType_INT48:
                switch (rank)
                {
                    case 0:
                        return new Tensor0<int64_t>(tensorName_, tensorDtype_, shape_);
                    case 1:
                        return new Tensor1<int64_t>(tensorName_, tensorDtype_, shape_);
                    case 2:
                        return new Tensor2<int64_t>(tensorName_, tensorDtype_, shape_);
                    case 3:
                        return new Tensor3<int64_t>(tensorName_, tensorDtype_, shape_);
                    case 4:
                        return new Tensor4<int64_t>(tensorName_, tensorDtype_, shape_);
                    case 5:
                        return new Tensor5<int64_t>(tensorName_, tensorDtype_, shape_);
                    case 6:
                        return new Tensor6<int64_t>(tensorName_, tensorDtype_, shape_);
                }
                break;
            case DType_BOOL:
                switch (rank)
                {
                    case 0:
                        return new Tensor0<bool>(tensorName_, tensorDtype_, shape_);
                    case 1:
                        return new Tensor1<bool>(tensorName_, tensorDtype_, shape_);
                    case 2:
                        return new Tensor2<bool>(tensorName_, tensorDtype_, shape_);
                    case 3:
                        return new Tensor3<bool>(tensorName_, tensorDtype_, shape_);
                    case 4:
                        return new Tensor4<bool>(tensorName_, tensorDtype_, shape_);
                    case 5:
                        return new Tensor5<bool>(tensorName_, tensorDtype_, shape_);
                    case 6:
                        return new Tensor6<bool>(tensorName_, tensorDtype_, shape_);
                }
                break;
            default:
                break;
        }
        return nullptr;
    }

    static Tensor* newTensor(DType type, const std::vector<int> shape);
};
};    // namespace TosaReference

#endif

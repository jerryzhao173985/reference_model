
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

#ifndef OP_TEMPLATE_TYPES_H
#define OP_TEMPLATE_TYPES_H

#include "tosa_generated.h"
#include <Eigen/CXX11/Tensor>

using namespace tosa;

namespace TosaReference
{
// Shorter alias templates for common Eigen::Tensor types
template <typename T>
using ETensor0 = Eigen::Tensor<T, 0>;
template <typename T>
using ETensor1 = Eigen::Tensor<T, 1>;
template <typename T>
using ETensor2 = Eigen::Tensor<T, 2>;
template <typename T>
using ETensor3 = Eigen::Tensor<T, 3>;
template <typename T>
using ETensor4 = Eigen::Tensor<T, 4>;
template <typename T>
using ETensor5 = Eigen::Tensor<T, 5>;
template <typename T>
using ETensor6 = Eigen::Tensor<T, 6>;

// Forward declaration
template <class T>
class TensorTemplate;

// Shortcut to hide the TensorTemplate class.
// For example, declare Tensor1<float> to get a TensorTemplate
// with an Eigen::Tensor<float, 1>
template <typename T>
using Tensor0 = TensorTemplate<ETensor0<T>>;
template <typename T>
using Tensor1 = TensorTemplate<ETensor1<T>>;
template <typename T>
using Tensor2 = TensorTemplate<ETensor2<T>>;
template <typename T>
using Tensor3 = TensorTemplate<ETensor3<T>>;
template <typename T>
using Tensor4 = TensorTemplate<ETensor4<T>>;
template <typename T>
using Tensor5 = TensorTemplate<ETensor5<T>>;
template <typename T>
using Tensor6 = TensorTemplate<ETensor6<T>>;

template <DType type>
struct GetEigenType;
template <>
struct GetEigenType<DType_FLOAT>
{
    using type = float;
};
template <>
struct GetEigenType<DType_INT32>
{
    using type = int32_t;
};
template <>
struct GetEigenType<DType_INT48>
{
    using type = int64_t;
};
template <>
struct GetEigenType<DType_BOOL>
{
    using type = bool;
};
template <>
struct GetEigenType<DType_UINT8>
{
    using type = int32_t;
};
template <>
struct GetEigenType<DType_UINT16>
{
    using type = int32_t;
};
template <>
struct GetEigenType<DType_INT4>
{
    using type = int32_t;
};
template <>
struct GetEigenType<DType_INT8>
{
    using type = int32_t;
};
template <>
struct GetEigenType<DType_INT16>
{
    using type = int32_t;
};

// Meta function to get number of bits
template <DType T>
struct GetNumBits
{
    static constexpr int32_t value = 0;
};
template <>
struct GetNumBits<DType_BOOL>
{
    static constexpr int32_t value = 1;
};
template <>
struct GetNumBits<DType_UINT8>
{
    static constexpr int32_t value = 8;
};
template <>
struct GetNumBits<DType_UINT16>
{
    static constexpr int32_t value = 16;
};
template <>
struct GetNumBits<DType_INT4>
{
    static constexpr int32_t value = 4;
};
template <>
struct GetNumBits<DType_INT8>
{
    static constexpr int32_t value = 8;
};
template <>
struct GetNumBits<DType_INT16>
{
    static constexpr int32_t value = 16;
};
template <>
struct GetNumBits<DType_INT32>
{
    static constexpr int32_t value = 32;
};
template <>
struct GetNumBits<DType_INT48>
{
    static constexpr int32_t value = 48;
};

// Meta function to get quantized min/max in compile time
template <DType T>
struct GetQMin
{
    static constexpr int64_t value = 0L;
};
template <>
struct GetQMin<DType_UINT8>
{
    static constexpr int64_t value = 0L;
};
template <>
struct GetQMin<DType_UINT16>
{
    static constexpr int64_t value = 0L;
};
template <>
struct GetQMin<DType_INT4>
{
    static constexpr int64_t value = -8L;
};
template <>
struct GetQMin<DType_INT8>
{
    static constexpr int64_t value = -128L;
};
template <>
struct GetQMin<DType_INT16>
{
    static constexpr int64_t value = -32768L;
};
template <>
struct GetQMin<DType_INT32>
{
    static constexpr int64_t value = -(1L << 31);
};
template <>
struct GetQMin<DType_INT48>
{
    static constexpr int64_t value = -(1L << 47);
};

template <DType T>
struct GetQMax
{
    static constexpr int64_t value = 0L;
};
template <>
struct GetQMax<DType_UINT8>
{
    static constexpr int64_t value = 255L;
};
template <>
struct GetQMax<DType_UINT16>
{
    static constexpr int64_t value = 65535L;
};
template <>
struct GetQMax<DType_INT4>
{
    static constexpr int64_t value = 7L;
};
template <>
struct GetQMax<DType_INT8>
{
    static constexpr int64_t value = 127L;
};
template <>
struct GetQMax<DType_INT16>
{
    static constexpr int64_t value = 32767L;
};
template <>
struct GetQMax<DType_INT32>
{
    static constexpr int64_t value = (1L << 31) - 1;
};
template <>
struct GetQMax<DType_INT48>
{
    static constexpr int64_t value = (1L << 47) - 1;
};

template <DType TIn1, DType TIn2>
struct GetAccDType;
template <>
struct GetAccDType<DType_INT8, DType_INT4>
{
    static constexpr DType value = DType_INT32;
};
template <>
struct GetAccDType<DType_INT8, DType_INT8>
{
    static constexpr DType value = DType_INT32;
};
template <>
struct GetAccDType<DType_INT16, DType_INT8>
{
    static constexpr DType value = DType_INT48;
};
template <>
struct GetAccDType<DType_INT16, DType_INT16>
{
    static constexpr DType value = DType_INT48;
};
template <>
struct GetAccDType<DType_FLOAT, DType_FLOAT>
{
    static constexpr DType value = DType_FLOAT;
};

};    // namespace TosaReference

#endif


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

#ifndef OP_TEMPLATE_TYPES_H
#define OP_TEMPLATE_TYPES_H

#include "tosa_generated.h"
#include <Eigen/CXX11/Tensor>
#include "half.hpp"
#include <Eigen/Core>
#include "arith_util.h"

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
struct GetEigenType<DType_FP32>
{
    using type = float;
};
template <>
struct GetEigenType<DType_FP16>
{
    // NOTE: full precision used
    using type = float;
};
template <>
struct GetEigenType<DType_BF16>
{
    // NOTE: full precision used
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

/* Get Accumulate Eigen Type:
Same behaviour as GetEigenType for all DTypes except the
single specialised case of DType_FP16. */
template <DType Dtype>
struct GetAccEigenType;
template <>
struct GetAccEigenType<DType_FP16>
{
    using type = half_float::half;
};
template <DType Dtype>
struct GetAccEigenType
{
    using type = typename GetEigenType<Dtype>::type;
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
template <>
struct GetNumBits<DType_FP16>
{
    static constexpr int32_t value = 16;
};

// Meta function to get quantized min/max in compile time
template <DType T>
struct GetQMin
{
    static constexpr int64_t value = INT64_C(0);
};
template <>
struct GetQMin<DType_UINT8>
{
    static constexpr int64_t value = INT64_C(0);
};
template <>
struct GetQMin<DType_UINT16>
{
    static constexpr int64_t value = INT64_C(0);
};
template <>
struct GetQMin<DType_INT4>
{
    static constexpr int64_t value = INT64_C(-8);
};
template <>
struct GetQMin<DType_INT8>
{
    static constexpr int64_t value = INT64_C(-128);
};
template <>
struct GetQMin<DType_INT16>
{
    static constexpr int64_t value = INT64_C(-32768);
};
template <>
struct GetQMin<DType_INT32>
{
    static constexpr int64_t value = -(INT64_C(1) << 31);
};
template <>
struct GetQMin<DType_INT48>
{
    static constexpr int64_t value = -(INT64_C(1) << 47);
};

template <DType T>
struct GetQMax
{
    static constexpr int64_t value = INT64_C(0);
};
template <>
struct GetQMax<DType_UINT8>
{
    static constexpr int64_t value = INT64_C(255);
};
template <>
struct GetQMax<DType_UINT16>
{
    static constexpr int64_t value = INT64_C(65535);
};
template <>
struct GetQMax<DType_INT4>
{
    static constexpr int64_t value = INT64_C(7);
};
template <>
struct GetQMax<DType_INT8>
{
    static constexpr int64_t value = INT64_C(127);
};
template <>
struct GetQMax<DType_INT16>
{
    static constexpr int64_t value = INT64_C(32767);
};
template <>
struct GetQMax<DType_INT32>
{
    static constexpr int64_t value = (INT64_C(1) << 31) - 1;
};
template <>
struct GetQMax<DType_INT48>
{
    static constexpr int64_t value = (INT64_C(1) << 47) - 1;
};

};    // namespace TosaReference

#endif

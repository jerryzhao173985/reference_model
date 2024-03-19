
// Copyright (c) 2020-2024, ARM Limited.
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

#ifndef OPS_TYPE_CONVERSION_H
#define OPS_TYPE_CONVERSION_H

#include "graph_node.h"

using namespace tosa;

namespace TosaReference
{
template <int Rank, TOSA_REF_TYPE InDtype, TOSA_REF_TYPE OutDtype>
class OpRescale : public GraphNode
{
public:
    OpRescale(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_);
    virtual ~OpRescale();

    virtual int checkTensorAttributes() final;
    virtual int eval() final;

    using InEigenType    = typename GetEigenType<InDtype>::type;
    using OutEigenType   = typename GetEigenType<OutDtype>::type;
    using TIn            = Eigen::Tensor<InEigenType, Rank>;
    using TOut           = Eigen::Tensor<OutEigenType, Rank>;
    using I8EigenType    = typename GetEigenType<TOSA_REF_TYPE::TOSA_REF_TYPE_INT8>::type;
    using I16EigenType   = typename GetEigenType<TOSA_REF_TYPE::TOSA_REF_TYPE_INT16>::type;
    using I32EigenType   = typename GetEigenType<TOSA_REF_TYPE::TOSA_REF_TYPE_INT32>::type;
    using TMultiplierI16 = Eigen::Tensor<I16EigenType, 1>;
    using TMultiplierI32 = Eigen::Tensor<I32EigenType, 1>;
    using TShift         = Eigen::Tensor<I8EigenType, 1>;

    bool isI8(TOSA_REF_TYPE Dtype)
    {
        return Dtype == TOSA_REF_TYPE_INT8 || Dtype == TOSA_REF_TYPE_UINT8;
    }

    bool isI16(TOSA_REF_TYPE Dtype)
    {
        return Dtype == TOSA_REF_TYPE_INT16 || Dtype == TOSA_REF_TYPE_UINT16;
    }

protected:
    TosaRescaleAttribute* attribute;
    TosaReference::TensorTemplate<TIn>* in;
    TosaReference::TensorTemplate<TOut>* out;
    TosaReference::TensorTemplate<TMultiplierI16>* multiplierI16;
    TosaReference::TensorTemplate<TMultiplierI32>* multiplierI32;
    TosaReference::TensorTemplate<TShift>* shift;
};

template <TOSA_REF_TYPE InDtype, TOSA_REF_TYPE OutDtype>
class CastHelper
{
public:
    using InEigenType                = typename GetEigenType<InDtype>::type;
    using OutEigenType               = typename GetEigenType<OutDtype>::type;
    using FcnType                    = std::function<OutEigenType(InEigenType)>;
    static constexpr int32_t OutBits = GetNumBits<OutDtype>::value;
    CastHelper();
    const FcnType& get_fcn() const
    {
        return fcn;
    }

private:
    FcnType fcn;
};

template <TOSA_REF_TYPE InDtype>
class CastHelper<InDtype, TOSA_REF_TYPE_BOOL>
{
public:
    using InEigenType  = typename GetEigenType<InDtype>::type;
    using OutEigenType = typename GetEigenType<TOSA_REF_TYPE_BOOL>::type;
    using FcnType      = std::function<OutEigenType(InEigenType)>;
    CastHelper();
    const FcnType& get_fcn() const
    {
        return fcn;
    }

private:
    FcnType fcn;
};

template <TOSA_REF_TYPE OutDtype>
class CastHelper<TOSA_REF_TYPE_BOOL, OutDtype>
{
public:
    using InEigenType               = typename GetEigenType<TOSA_REF_TYPE_BOOL>::type;
    using OutEigenType              = typename GetEigenType<OutDtype>::type;
    using FcnType                   = std::function<OutEigenType(InEigenType)>;
    static constexpr int32_t OutMin = GetQMin<OutDtype>::value;
    static constexpr int32_t OutMax = GetQMax<OutDtype>::value;
    CastHelper();
    const FcnType& get_fcn() const
    {
        return fcn;
    }

private:
    FcnType fcn;
};

template <TOSA_REF_TYPE InDtype>
class CastHelper<InDtype, TOSA_REF_TYPE_FP16>
{
public:
    using InEigenType  = typename GetEigenType<InDtype>::type;
    using OutEigenType = typename GetEigenType<TOSA_REF_TYPE_FP16>::type;
    using FcnType      = std::function<OutEigenType(InEigenType)>;
    CastHelper();
    const FcnType& get_fcn() const
    {
        return fcn;
    }

private:
    FcnType fcn;
};

template <TOSA_REF_TYPE OutDtype>
class CastHelper<TOSA_REF_TYPE_FP16, OutDtype>
{
public:
    using InEigenType               = typename GetEigenType<TOSA_REF_TYPE_FP16>::type;
    using OutEigenType              = typename GetEigenType<OutDtype>::type;
    using FcnType                   = std::function<OutEigenType(InEigenType)>;
    static constexpr int32_t OutMin = GetQMin<OutDtype>::value;
    static constexpr int32_t OutMax = GetQMax<OutDtype>::value;
    CastHelper();
    const FcnType& get_fcn() const
    {
        return fcn;
    }

private:
    FcnType fcn;
};

template <>
class CastHelper<TOSA_REF_TYPE_FP32, TOSA_REF_TYPE_FP16>
{
public:
    using InEigenType  = typename GetEigenType<TOSA_REF_TYPE_FP32>::type;
    using OutEigenType = typename GetEigenType<TOSA_REF_TYPE_FP16>::type;
    using FcnType      = std::function<OutEigenType(InEigenType)>;
    CastHelper();
    const FcnType& get_fcn() const
    {
        return fcn;
    }

private:
    FcnType fcn;
};

template <TOSA_REF_TYPE InDtype>
class CastHelper<InDtype, TOSA_REF_TYPE_BF16>
{
public:
    using InEigenType  = typename GetEigenType<InDtype>::type;
    using OutEigenType = typename GetEigenType<TOSA_REF_TYPE_BF16>::type;
    using FcnType      = std::function<OutEigenType(InEigenType)>;
    CastHelper();
    const FcnType& get_fcn() const
    {
        return fcn;
    }

private:
    FcnType fcn;
};

template <TOSA_REF_TYPE OutDtype>
class CastHelper<TOSA_REF_TYPE_BF16, OutDtype>
{
public:
    using InEigenType               = typename GetEigenType<TOSA_REF_TYPE_BF16>::type;
    using OutEigenType              = typename GetEigenType<OutDtype>::type;
    using FcnType                   = std::function<OutEigenType(InEigenType)>;
    static constexpr int32_t OutMin = GetQMin<OutDtype>::value;
    static constexpr int32_t OutMax = GetQMax<OutDtype>::value;
    CastHelper();
    const FcnType& get_fcn() const
    {
        return fcn;
    }

private:
    FcnType fcn;
};

template <>
class CastHelper<TOSA_REF_TYPE_FP32, TOSA_REF_TYPE_BF16>
{
public:
    using InEigenType  = typename GetEigenType<TOSA_REF_TYPE_FP32>::type;
    using OutEigenType = typename GetEigenType<TOSA_REF_TYPE_BF16>::type;
    using FcnType      = std::function<OutEigenType(InEigenType)>;
    CastHelper();
    const FcnType& get_fcn() const
    {
        return fcn;
    }

private:
    FcnType fcn;
};

template <TOSA_REF_TYPE InDtype>
class CastHelper<InDtype, TOSA_REF_TYPE_FP32>
{
public:
    using InEigenType  = typename GetEigenType<InDtype>::type;
    using OutEigenType = typename GetEigenType<TOSA_REF_TYPE_FP32>::type;
    using FcnType      = std::function<OutEigenType(InEigenType)>;
    CastHelper();
    const FcnType& get_fcn() const
    {
        return fcn;
    }

private:
    FcnType fcn;
};

template <>
class CastHelper<TOSA_REF_TYPE_FP16, TOSA_REF_TYPE_FP32>
{
public:
    using InEigenType  = typename GetEigenType<TOSA_REF_TYPE_FP16>::type;
    using OutEigenType = typename GetEigenType<TOSA_REF_TYPE_FP32>::type;
    using FcnType      = std::function<OutEigenType(InEigenType)>;
    CastHelper();
    const FcnType& get_fcn() const
    {
        return fcn;
    }

private:
    FcnType fcn;
};

template <>
class CastHelper<TOSA_REF_TYPE_BF16, TOSA_REF_TYPE_FP32>
{
public:
    using InEigenType  = typename GetEigenType<TOSA_REF_TYPE_BF16>::type;
    using OutEigenType = typename GetEigenType<TOSA_REF_TYPE_FP32>::type;
    using FcnType      = std::function<OutEigenType(InEigenType)>;
    CastHelper();
    const FcnType& get_fcn() const
    {
        return fcn;
    }

private:
    FcnType fcn;
};

template <TOSA_REF_TYPE OutDtype>
class CastHelper<TOSA_REF_TYPE_FP32, OutDtype>
{
public:
    using InEigenType               = typename GetEigenType<TOSA_REF_TYPE_FP32>::type;
    using OutEigenType              = typename GetEigenType<OutDtype>::type;
    using FcnType                   = std::function<OutEigenType(InEigenType)>;
    static constexpr int32_t OutMin = GetQMin<OutDtype>::value;
    static constexpr int32_t OutMax = GetQMax<OutDtype>::value;
    CastHelper();
    const FcnType& get_fcn() const
    {
        return fcn;
    }

private:
    FcnType fcn;
};

template <TOSA_REF_TYPE OutDtype>
class CastHelper<TOSA_REF_TYPE_FP8E4M3, OutDtype>
{
public:
    using InEigenType               = typename GetEigenType<TOSA_REF_TYPE_FP8E4M3>::type;
    using OutEigenType              = typename GetEigenType<OutDtype>::type;
    using FcnType                   = std::function<OutEigenType(InEigenType)>;
    static constexpr int32_t OutMin = GetQMin<OutDtype>::value;
    static constexpr int32_t OutMax = GetQMax<OutDtype>::value;
    CastHelper();
    const FcnType& get_fcn() const
    {
        return fcn;
    }

private:
    FcnType fcn;
};

template <>
class CastHelper<TOSA_REF_TYPE_FP8E4M3, TOSA_REF_TYPE_FP16>
{
public:
    using InEigenType  = typename GetEigenType<TOSA_REF_TYPE_FP8E4M3>::type;
    using OutEigenType = typename GetEigenType<TOSA_REF_TYPE_FP16>::type;
    using FcnType      = std::function<OutEigenType(InEigenType)>;
    CastHelper();
    const FcnType& get_fcn() const
    {
        return fcn;
    }

private:
    FcnType fcn;
};

template <>
class CastHelper<TOSA_REF_TYPE_FP8E4M3, TOSA_REF_TYPE_BF16>
{
public:
    using InEigenType  = typename GetEigenType<TOSA_REF_TYPE_FP8E4M3>::type;
    using OutEigenType = typename GetEigenType<TOSA_REF_TYPE_BF16>::type;
    using FcnType      = std::function<OutEigenType(InEigenType)>;
    CastHelper();
    const FcnType& get_fcn() const
    {
        return fcn;
    }

private:
    FcnType fcn;
};

template <>
class CastHelper<TOSA_REF_TYPE_FP8E4M3, TOSA_REF_TYPE_FP32>
{
public:
    using InEigenType  = typename GetEigenType<TOSA_REF_TYPE_FP8E4M3>::type;
    using OutEigenType = typename GetEigenType<TOSA_REF_TYPE_FP32>::type;
    using FcnType      = std::function<OutEigenType(InEigenType)>;
    CastHelper();
    const FcnType& get_fcn() const
    {
        return fcn;
    }

private:
    FcnType fcn;
};

template <TOSA_REF_TYPE OutDtype>
class CastHelper<TOSA_REF_TYPE_FP8E5M2, OutDtype>
{
public:
    using InEigenType               = typename GetEigenType<TOSA_REF_TYPE_FP8E5M2>::type;
    using OutEigenType              = typename GetEigenType<OutDtype>::type;
    using FcnType                   = std::function<OutEigenType(InEigenType)>;
    static constexpr int32_t OutMin = GetQMin<OutDtype>::value;
    static constexpr int32_t OutMax = GetQMax<OutDtype>::value;
    CastHelper();
    const FcnType& get_fcn() const
    {
        return fcn;
    }

private:
    FcnType fcn;
};

template <>
class CastHelper<TOSA_REF_TYPE_FP8E5M2, TOSA_REF_TYPE_FP16>
{
public:
    using InEigenType  = typename GetEigenType<TOSA_REF_TYPE_FP8E5M2>::type;
    using OutEigenType = typename GetEigenType<TOSA_REF_TYPE_FP16>::type;
    using FcnType      = std::function<OutEigenType(InEigenType)>;
    CastHelper();
    const FcnType& get_fcn() const
    {
        return fcn;
    }

private:
    FcnType fcn;
};

template <>
class CastHelper<TOSA_REF_TYPE_FP8E5M2, TOSA_REF_TYPE_BF16>
{
public:
    using InEigenType  = typename GetEigenType<TOSA_REF_TYPE_FP8E5M2>::type;
    using OutEigenType = typename GetEigenType<TOSA_REF_TYPE_BF16>::type;
    using FcnType      = std::function<OutEigenType(InEigenType)>;
    CastHelper();
    const FcnType& get_fcn() const
    {
        return fcn;
    }

private:
    FcnType fcn;
};

template <>
class CastHelper<TOSA_REF_TYPE_FP8E5M2, TOSA_REF_TYPE_FP32>
{
public:
    using InEigenType  = typename GetEigenType<TOSA_REF_TYPE_FP8E5M2>::type;
    using OutEigenType = typename GetEigenType<TOSA_REF_TYPE_FP32>::type;
    using FcnType      = std::function<OutEigenType(InEigenType)>;
    CastHelper();
    const FcnType& get_fcn() const
    {
        return fcn;
    }

private:
    FcnType fcn;
};

template <TOSA_REF_TYPE InDtype>
class CastHelper<InDtype, TOSA_REF_TYPE_FP8E4M3>
{
public:
    using InEigenType  = typename GetEigenType<InDtype>::type;
    using OutEigenType = typename GetEigenType<TOSA_REF_TYPE_FP8E4M3>::type;
    using FcnType      = std::function<OutEigenType(InEigenType)>;
    CastHelper();
    const FcnType& get_fcn() const
    {
        return fcn;
    }

private:
    FcnType fcn;
};

template <>
class CastHelper<TOSA_REF_TYPE_FP16, TOSA_REF_TYPE_FP8E4M3>
{
public:
    using InEigenType  = typename GetEigenType<TOSA_REF_TYPE_FP16>::type;
    using OutEigenType = typename GetEigenType<TOSA_REF_TYPE_FP8E4M3>::type;
    using FcnType      = std::function<OutEigenType(InEigenType)>;
    CastHelper();
    const FcnType& get_fcn() const
    {
        return fcn;
    }

private:
    FcnType fcn;
};

template <>
class CastHelper<TOSA_REF_TYPE_BF16, TOSA_REF_TYPE_FP8E4M3>
{
public:
    using InEigenType  = typename GetEigenType<TOSA_REF_TYPE_BF16>::type;
    using OutEigenType = typename GetEigenType<TOSA_REF_TYPE_FP8E4M3>::type;
    using FcnType      = std::function<OutEigenType(InEigenType)>;
    CastHelper();
    const FcnType& get_fcn() const
    {
        return fcn;
    }

private:
    FcnType fcn;
};

template <>
class CastHelper<TOSA_REF_TYPE_FP32, TOSA_REF_TYPE_FP8E4M3>
{
public:
    using InEigenType  = typename GetEigenType<TOSA_REF_TYPE_FP32>::type;
    using OutEigenType = typename GetEigenType<TOSA_REF_TYPE_FP8E4M3>::type;
    using FcnType      = std::function<OutEigenType(InEigenType)>;
    CastHelper();
    const FcnType& get_fcn() const
    {
        return fcn;
    }

private:
    FcnType fcn;
};

template <TOSA_REF_TYPE InDtype>
class CastHelper<InDtype, TOSA_REF_TYPE_FP8E5M2>
{
public:
    using InEigenType  = typename GetEigenType<InDtype>::type;
    using OutEigenType = typename GetEigenType<TOSA_REF_TYPE_FP8E5M2>::type;
    using FcnType      = std::function<OutEigenType(InEigenType)>;
    CastHelper();
    const FcnType& get_fcn() const
    {
        return fcn;
    }

private:
    FcnType fcn;
};

template <>
class CastHelper<TOSA_REF_TYPE_FP16, TOSA_REF_TYPE_FP8E5M2>
{
public:
    using InEigenType  = typename GetEigenType<TOSA_REF_TYPE_FP16>::type;
    using OutEigenType = typename GetEigenType<TOSA_REF_TYPE_FP8E5M2>::type;
    using FcnType      = std::function<OutEigenType(InEigenType)>;
    CastHelper();
    const FcnType& get_fcn() const
    {
        return fcn;
    }

private:
    FcnType fcn;
};

template <>
class CastHelper<TOSA_REF_TYPE_BF16, TOSA_REF_TYPE_FP8E5M2>
{
public:
    using InEigenType  = typename GetEigenType<TOSA_REF_TYPE_BF16>::type;
    using OutEigenType = typename GetEigenType<TOSA_REF_TYPE_FP8E5M2>::type;
    using FcnType      = std::function<OutEigenType(InEigenType)>;
    CastHelper();
    const FcnType& get_fcn() const
    {
        return fcn;
    }

private:
    FcnType fcn;
};

template <>
class CastHelper<TOSA_REF_TYPE_FP32, TOSA_REF_TYPE_FP8E5M2>
{
public:
    using InEigenType  = typename GetEigenType<TOSA_REF_TYPE_FP32>::type;
    using OutEigenType = typename GetEigenType<TOSA_REF_TYPE_FP8E5M2>::type;
    using FcnType      = std::function<OutEigenType(InEigenType)>;
    CastHelper();
    const FcnType& get_fcn() const
    {
        return fcn;
    }

private:
    FcnType fcn;
};

template <TOSA_REF_TYPE OutDtype>
class CastHelper<TOSA_REF_TYPE_FP64, OutDtype>
{
public:
    using InEigenType               = typename GetEigenType<TOSA_REF_TYPE_FP64>::type;
    using OutEigenType              = typename GetEigenType<OutDtype>::type;
    using FcnType                   = std::function<OutEigenType(InEigenType)>;
    static constexpr int32_t OutMin = GetQMin<OutDtype>::value;
    static constexpr int32_t OutMax = GetQMax<OutDtype>::value;
    CastHelper();
    const FcnType& get_fcn() const
    {
        return fcn;
    }

private:
    FcnType fcn;
};

template <int Rank, TOSA_REF_TYPE InDtype, TOSA_REF_TYPE OutDtype>
class OpCast : public GraphNode
{
public:
    OpCast(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_);
    virtual ~OpCast();

    virtual int checkTensorAttributes() final;
    virtual int eval() final;

    using InEigenType  = typename GetEigenType<InDtype>::type;
    using OutEigenType = typename GetEigenType<OutDtype>::type;
    using TIn          = Eigen::Tensor<InEigenType, Rank>;
    using TOut         = Eigen::Tensor<OutEigenType, Rank>;

protected:
    CastHelper<InDtype, OutDtype> cast_helper;
    TosaReference::TensorTemplate<TIn>* in;
    TosaReference::TensorTemplate<TOut>* out;
};

};    // namespace TosaReference

#endif

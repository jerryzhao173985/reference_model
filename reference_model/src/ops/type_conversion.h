
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

#ifndef OPS_TYPE_CONVERSION_H
#define OPS_TYPE_CONVERSION_H

#include "graph_node.h"

using namespace tosa;

namespace TosaReference
{
template <int Rank, DType InDtype, DType OutDtype>
class OpRescale : public GraphNode
{
public:
    OpRescale(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_);
    virtual ~OpRescale();

    virtual int checkTensorAttributes() final;
    virtual int eval() final;

    using InEigenType  = typename GetEigenType<InDtype>::type;
    using OutEigenType = typename GetEigenType<OutDtype>::type;
    using TIn          = Eigen::Tensor<InEigenType, Rank>;
    using TOut         = Eigen::Tensor<OutEigenType, Rank>;

    static constexpr int32_t QMin = GetQMin<OutDtype>::value;
    static constexpr int32_t QMax = GetQMax<OutDtype>::value;

protected:
    TosaRescaleAttribute* attribute;
    TosaReference::TensorTemplate<TIn>* in;
    TosaReference::TensorTemplate<TOut>* out;
};

template <DType InDtype, DType OutDtype>
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

template <DType InDtype>
class CastHelper<InDtype, DType_BOOL>
{
public:
    using InEigenType  = typename GetEigenType<InDtype>::type;
    using OutEigenType = typename GetEigenType<DType_BOOL>::type;
    using FcnType      = std::function<OutEigenType(InEigenType)>;
    CastHelper();
    const FcnType& get_fcn() const
    {
        return fcn;
    }

private:
    FcnType fcn;
};

template <DType OutDtype>
class CastHelper<DType_BOOL, OutDtype>
{
public:
    using InEigenType               = typename GetEigenType<DType_BOOL>::type;
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

template <DType InDtype>
class CastHelper<InDtype, DType_FP16>
{
public:
    using InEigenType  = typename GetEigenType<InDtype>::type;
    using OutEigenType = typename GetEigenType<DType_FP16>::type;
    using FcnType      = std::function<OutEigenType(InEigenType)>;
    CastHelper();
    const FcnType& get_fcn() const
    {
        return fcn;
    }

private:
    FcnType fcn;
};

template <DType OutDtype>
class CastHelper<DType_FP16, OutDtype>
{
public:
    using InEigenType               = typename GetEigenType<DType_FP16>::type;
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

template <DType InDtype>
class CastHelper<InDtype, DType_FLOAT>
{
public:
    using InEigenType  = typename GetEigenType<InDtype>::type;
    using OutEigenType = typename GetEigenType<DType_FLOAT>::type;
    using FcnType      = std::function<OutEigenType(InEigenType)>;
    CastHelper();
    const FcnType& get_fcn() const
    {
        return fcn;
    }

private:
    FcnType fcn;
};

template <DType OutDtype>
class CastHelper<DType_FLOAT, OutDtype>
{
public:
    using InEigenType               = typename GetEigenType<DType_FLOAT>::type;
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

template <int Rank, DType InDtype, DType OutDtype>
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


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

#ifndef OPS_TENSOR_OPS_H
#define OPS_TENSOR_OPS_H

#include "graph_node.h"
#include "quant_util.h"

using namespace tosa;

namespace TosaReference
{

template <int Rank, TOSA_REF_TYPE Dtype>
class OpArgMax : public GraphNode
{
public:
    OpArgMax(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_);
    virtual ~OpArgMax();

    virtual int checkTensorAttributes();
    virtual int eval();

    using InEigenType  = typename GetEigenType<Dtype>::type;
    using OutEigenType = typename GetEigenType<TOSA_REF_TYPE_INT32>::type;
    using TIn          = Eigen::Tensor<InEigenType, Rank>;
    using TOut         = Eigen::Tensor<OutEigenType, Rank - 1>;

protected:
    TosaAxisAttribute* attribute;
    TosaReference::TensorTemplate<TIn>* input;
    TosaReference::TensorTemplate<TOut>* output;
};

template <TOSA_REF_TYPE Dtype, TOSA_REF_TYPE AccDtype>
class OpAvgPool2d : public GraphNode
{
public:
    OpAvgPool2d(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_);
    virtual ~OpAvgPool2d();

    virtual int checkTensorAttributes();
    virtual int eval();

    using InEigenType  = typename GetEigenType<Dtype>::type;
    using AccEigenType = typename GetAccEigenType<AccDtype>::type;    // Note: different from GetEigenType
    using OutEigenType = typename GetEigenType<Dtype>::type;
    using TIn          = Eigen::Tensor<InEigenType, 4>;
    using TOut         = Eigen::Tensor<OutEigenType, 4>;

    static constexpr int64_t QMin = GetQMin<Dtype>::value;
    static constexpr int64_t QMax = GetQMax<Dtype>::value;

protected:
    TosaReference::TensorTemplate<TIn>* in;
    TosaReference::TensorTemplate<TOut>* out;
    tosa::TosaPoolAttribute* attribute;

protected:
    // return a 1D [N] tensor that describes a how many valid elements covered in the input space
    ETensor1<int32_t> calculate_div_map_1d(
        int in_size, int out_size, int kernel_size, int stride, int32_t padding_left, int32_t padding_right);
};

template <TOSA_REF_TYPE InDtype, TOSA_REF_TYPE WeightDtype, TOSA_REF_TYPE AccDtype, TOSA_REF_TYPE OutDtype>
class OpConv2d : public GraphNode
{
public:
    OpConv2d(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_);
    virtual ~OpConv2d();

    virtual int checkTensorAttributes() final;
    virtual int eval() final;

    using InEigenType     = typename GetEigenType<InDtype>::type;
    using WeightEigenType = typename GetEigenType<WeightDtype>::type;
    using AccEigenType    = typename GetAccEigenType<AccDtype>::type;    // Note: different from GetEigenType
    using OutEigenType    = typename GetEigenType<OutDtype>::type;
    using TIn             = Eigen::Tensor<InEigenType, 4>;
    using TWeight         = Eigen::Tensor<WeightEigenType, 4>;
    using TBias           = Eigen::Tensor<OutEigenType, 1>;
    using TOut            = Eigen::Tensor<OutEigenType, 4>;

    static constexpr int64_t AccQMin = GetQMin<OutDtype>::value;
    static constexpr int64_t AccQMax = GetQMax<OutDtype>::value;

protected:
    TosaReference::TensorTemplate<TIn>* input;
    TosaReference::TensorTemplate<TWeight>* weight;
    TosaReference::TensorTemplate<TBias>* bias;
    TosaReference::TensorTemplate<TOut>* output;
    tosa::TosaConvAttribute* attribute;
};

template <TOSA_REF_TYPE InDtype, TOSA_REF_TYPE WeightDtype, TOSA_REF_TYPE AccDtype, TOSA_REF_TYPE OutDtype>
class OpConv3d : public GraphNode
{
public:
    OpConv3d(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_);
    virtual ~OpConv3d();

    virtual int checkTensorAttributes() final;
    virtual int eval() final;

    using InEigenType     = typename GetEigenType<InDtype>::type;
    using WeightEigenType = typename GetEigenType<WeightDtype>::type;
    using AccEigenType    = typename GetAccEigenType<AccDtype>::type;    // Note: different from GetEigenType
    using OutEigenType    = typename GetEigenType<OutDtype>::type;
    using TIn             = Eigen::Tensor<InEigenType, 5>;
    using TWeight         = Eigen::Tensor<WeightEigenType, 5>;
    using TBias           = Eigen::Tensor<OutEigenType, 1>;
    using TOut            = Eigen::Tensor<OutEigenType, 5>;

    static constexpr int64_t AccQMin = GetQMin<OutDtype>::value;
    static constexpr int64_t AccQMax = GetQMax<OutDtype>::value;

protected:
    TosaReference::TensorTemplate<TIn>* input;
    TosaReference::TensorTemplate<TWeight>* weight;
    TosaReference::TensorTemplate<TBias>* bias;
    TosaReference::TensorTemplate<TOut>* output;
    tosa::TosaConvAttribute* attribute;
};

template <TOSA_REF_TYPE InDtype, TOSA_REF_TYPE WeightDtype, TOSA_REF_TYPE AccDtype, TOSA_REF_TYPE OutDtype>
class OpDepthwiseConv2d : public GraphNode
{
public:
    OpDepthwiseConv2d(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_);
    virtual ~OpDepthwiseConv2d();

    virtual int checkTensorAttributes() final;
    virtual int eval() final;

    using InEigenType     = typename GetEigenType<InDtype>::type;
    using WeightEigenType = typename GetEigenType<WeightDtype>::type;
    using AccEigenType    = typename GetAccEigenType<AccDtype>::type;    // Note: different from GetEigenType
    using OutEigenType    = typename GetEigenType<OutDtype>::type;
    using TIn             = Eigen::Tensor<InEigenType, 4>;
    using TWeight         = Eigen::Tensor<WeightEigenType, 4>;
    using TBias           = Eigen::Tensor<OutEigenType, 1>;
    using TOut            = Eigen::Tensor<OutEigenType, 4>;

    static constexpr int64_t AccQMin = GetQMin<OutDtype>::value;
    static constexpr int64_t AccQMax = GetQMax<OutDtype>::value;

protected:
    TosaReference::TensorTemplate<TIn>* input;
    TosaReference::TensorTemplate<TWeight>* weight;
    TosaReference::TensorTemplate<TBias>* bias;
    TosaReference::TensorTemplate<TOut>* output;
    tosa::TosaConvAttribute* attribute;
};

template <TOSA_REF_TYPE InDtype, TOSA_REF_TYPE WeightDtype, TOSA_REF_TYPE OutDtype>
class OpFullyConnected : public GraphNode
{
public:
    OpFullyConnected(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_);
    virtual ~OpFullyConnected();

    virtual int checkTensorAttributes() final;
    virtual int eval() final;

    using InEigenType     = typename GetEigenType<InDtype>::type;
    using WeightEigenType = typename GetEigenType<WeightDtype>::type;
    using AccEigenType    = typename GetAccEigenType<OutDtype>::type;    // Note: different from GetEigenType
    using OutEigenType    = typename GetEigenType<OutDtype>::type;
    using TIn             = Eigen::Tensor<InEigenType, 2>;
    using TWeight         = Eigen::Tensor<WeightEigenType, 2>;
    using TBias           = Eigen::Tensor<OutEigenType, 1>;
    using TOut            = Eigen::Tensor<OutEigenType, 2>;

    static constexpr int64_t AccQMin = GetQMin<OutDtype>::value;
    static constexpr int64_t AccQMax = GetQMax<OutDtype>::value;

protected:
    TosaReference::TensorTemplate<TIn>* input;
    TosaReference::TensorTemplate<TWeight>* weight;
    TosaReference::TensorTemplate<TBias>* bias;
    TosaReference::TensorTemplate<TOut>* output;

    tosa::TosaFullyConnectedAttribute* attribute;
};

template <TOSA_REF_TYPE Dtype, TOSA_REF_TYPE OutDtype>
class OpMatMul : public GraphNode
{
public:
    OpMatMul(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_);
    virtual ~OpMatMul();

    virtual int checkTensorAttributes() final;
    virtual int eval() final;

    using InEigenType                = typename GetEigenType<Dtype>::type;
    using AccEigenType               = typename GetAccEigenType<OutDtype>::type;    // Note: different from GetEigenType
    using OutEigenType               = typename GetEigenType<OutDtype>::type;
    using TIn                        = Eigen::Tensor<InEigenType, 3>;
    using TOut                       = Eigen::Tensor<OutEigenType, 3>;
    using TInRank2                   = Eigen::Tensor<InEigenType, 2>;
    using TAccRank2                  = Eigen::Tensor<AccEigenType, 2>;
    static constexpr int64_t AccQMin = GetQMin<OutDtype>::value;
    static constexpr int64_t AccQMax = GetQMax<OutDtype>::value;

protected:
    TosaReference::TensorTemplate<TIn>* a;
    TosaReference::TensorTemplate<TIn>* b;
    TosaReference::TensorTemplate<TOut>* output;
    int64_t N;
    int64_t H;
    int64_t W;
    int64_t C;

    tosa::TosaMatMulAttribute* attribute;
};

template <TOSA_REF_TYPE Dtype>
class OpMaxPool2d : public GraphNode
{
public:
    OpMaxPool2d(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_);
    virtual ~OpMaxPool2d();

    virtual int checkTensorAttributes();
    virtual int eval();

    using InEigenType  = typename GetEigenType<Dtype>::type;
    using OutEigenType = typename GetEigenType<Dtype>::type;
    using TIn          = Eigen::Tensor<InEigenType, 4>;
    using TOut         = Eigen::Tensor<OutEigenType, 4>;

protected:
    TosaReference::TensorTemplate<TIn>* in;
    TosaReference::TensorTemplate<TOut>* out;
    tosa::TosaPoolAttribute* attribute;
};

template <TOSA_REF_TYPE Dtype>
class OpFFT2d : public GraphNode
{
public:
    OpFFT2d(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_);
    virtual ~OpFFT2d();

    virtual int checkTensorAttributes() final;
    virtual int eval() final;

    using InEigenType  = typename GetEigenType<Dtype>::type;
    using OutEigenType = typename GetEigenType<Dtype>::type;
    using TIn          = Eigen::Tensor<InEigenType, 3>;
    using TOut         = Eigen::Tensor<OutEigenType, 3>;

protected:
    TosaReference::TensorTemplate<TIn>* in_real;
    TosaReference::TensorTemplate<TIn>* in_imag;
    TosaReference::TensorTemplate<TOut>* out_real;
    TosaReference::TensorTemplate<TOut>* out_imag;
    tosa::TosaFFTAttribute* attribute;
};

template <TOSA_REF_TYPE Dtype>
class OpRFFT2d : public GraphNode
{
public:
    OpRFFT2d(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_);
    virtual ~OpRFFT2d();

    virtual int checkTensorAttributes() final;
    virtual int eval() final;

    using InEigenType  = typename GetEigenType<Dtype>::type;
    using OutEigenType = typename GetEigenType<Dtype>::type;
    using TIn          = Eigen::Tensor<InEigenType, 3>;
    using TOut         = Eigen::Tensor<OutEigenType, 3>;

protected:
    TosaReference::TensorTemplate<TIn>* in;
    TosaReference::TensorTemplate<TOut>* out_real;
    TosaReference::TensorTemplate<TOut>* out_imag;
    tosa::TosaRFFTAttribute* attribute;
};

template <TOSA_REF_TYPE InDtype, TOSA_REF_TYPE WeightDtype, TOSA_REF_TYPE AccDtype, TOSA_REF_TYPE OutDtype>
class OpTransposeConv2d : public GraphNode
{
public:
    OpTransposeConv2d(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_);
    virtual ~OpTransposeConv2d();

    virtual int checkTensorAttributes() final;
    virtual int eval() final;

    using InEigenType     = typename GetEigenType<InDtype>::type;
    using WeightEigenType = typename GetEigenType<WeightDtype>::type;
    using AccEigenType    = typename GetAccEigenType<AccDtype>::type;    // Note: different from GetEigenType
    using OutEigenType    = typename GetEigenType<OutDtype>::type;
    using TIn             = Eigen::Tensor<InEigenType, 4>;
    using TWeight         = Eigen::Tensor<WeightEigenType, 4>;
    using TBias           = Eigen::Tensor<OutEigenType, 1>;
    using TOut            = Eigen::Tensor<OutEigenType, 4>;

    static constexpr int64_t AccQMin = GetQMin<OutDtype>::value;
    static constexpr int64_t AccQMax = GetQMax<OutDtype>::value;

protected:
    TosaReference::TensorTemplate<TIn>* input;
    TosaReference::TensorTemplate<TWeight>* weight;
    TosaReference::TensorTemplate<TBias>* bias;
    TosaReference::TensorTemplate<TOut>* output;
    TosaTransposeConvAttribute* attribute;
};

};    // namespace TosaReference

#endif

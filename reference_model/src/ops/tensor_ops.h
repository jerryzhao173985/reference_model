
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
    std::unique_ptr<tosa::TosaArgMaxAttribute> attribute;
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
    using TInZp        = Eigen::Tensor<InEigenType, 1>;
    using TOutZp       = Eigen::Tensor<OutEigenType, 1>;

    static constexpr int64_t QMin = GetQMin<Dtype>::value;
    static constexpr int64_t QMax = GetQMax<Dtype>::value;

protected:
    TosaReference::TensorTemplate<TIn>* in;
    TosaReference::TensorTemplate<TOut>* out;
    TosaReference::TensorTemplate<TInZp>* input_zp;
    TosaReference::TensorTemplate<TOutZp>* output_zp;
    std::unique_ptr<tosa::TosaAvgPool2dAttribute> attribute;
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
    using TInZp           = Eigen::Tensor<InEigenType, 1>;
    using TWeightZp       = Eigen::Tensor<WeightEigenType, 1>;
    using TOut            = Eigen::Tensor<OutEigenType, 4>;

    static constexpr int64_t AccQMin = GetQMin<OutDtype>::value;
    static constexpr int64_t AccQMax = GetQMax<OutDtype>::value;

protected:
    TosaReference::TensorTemplate<TIn>* input;
    TosaReference::TensorTemplate<TWeight>* weight;
    TosaReference::TensorTemplate<TBias>* bias;
    TosaReference::TensorTemplate<TOut>* output;
    std::unique_ptr<tosa::TosaConv2dAttribute> attribute;
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
    using TInZp           = Eigen::Tensor<InEigenType, 1>;
    using TWeightZp       = Eigen::Tensor<WeightEigenType, 1>;
    using TOut            = Eigen::Tensor<OutEigenType, 5>;

    static constexpr int64_t AccQMin = GetQMin<OutDtype>::value;
    static constexpr int64_t AccQMax = GetQMax<OutDtype>::value;

protected:
    TosaReference::TensorTemplate<TIn>* input;
    TosaReference::TensorTemplate<TWeight>* weight;
    TosaReference::TensorTemplate<TBias>* bias;
    TosaReference::TensorTemplate<TOut>* output;
    std::unique_ptr<tosa::TosaConv3dAttribute> attribute;
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
    using TInZp           = Eigen::Tensor<InEigenType, 1>;
    using TWeightZp       = Eigen::Tensor<WeightEigenType, 1>;
    using TBias           = Eigen::Tensor<OutEigenType, 1>;
    using TOut            = Eigen::Tensor<OutEigenType, 4>;

    static constexpr int64_t AccQMin = GetQMin<OutDtype>::value;
    static constexpr int64_t AccQMax = GetQMax<OutDtype>::value;

protected:
    TosaReference::TensorTemplate<TIn>* input;
    TosaReference::TensorTemplate<TWeight>* weight;
    TosaReference::TensorTemplate<TBias>* bias;
    TosaReference::TensorTemplate<TOut>* output;
    std::unique_ptr<tosa::TosaDepthwiseConv2dAttribute> attribute;
};

template <TOSA_REF_TYPE InputADtype, TOSA_REF_TYPE InputBDtype, TOSA_REF_TYPE OutDtype>
class OpMatMul : public GraphNode
{
public:
    OpMatMul(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_);

    virtual int checkTensorAttributes() final;
    virtual int eval() final;

    using InputAEigenType            = typename GetEigenType<InputADtype>::type;
    using InputBEigenType            = typename GetEigenType<InputBDtype>::type;
    using AccEigenType               = typename GetAccEigenType<OutDtype>::type;    // Note: different from GetEigenType
    using OutEigenType               = typename GetEigenType<OutDtype>::type;
    using TInputA                    = Eigen::Tensor<InputAEigenType, 3>;
    using TInputB                    = Eigen::Tensor<InputBEigenType, 3>;
    using TZeroPointA                = Eigen::Tensor<InputBEigenType, 1>;
    using TZeroPointB                = Eigen::Tensor<InputBEigenType, 1>;
    using TOut                       = Eigen::Tensor<OutEigenType, 3>;
    using TInput1Rank2               = Eigen::Tensor<InputAEigenType, 2>;
    using TInput2Rank2               = Eigen::Tensor<InputBEigenType, 2>;
    using TAccRank2                  = Eigen::Tensor<AccEigenType, 2>;
    static constexpr int64_t AccQMin = GetQMin<OutDtype>::value;
    static constexpr int64_t AccQMax = GetQMax<OutDtype>::value;

protected:
    TosaReference::TensorTemplate<TInputA>* a;
    TosaReference::TensorTemplate<TInputB>* b;
    TosaReference::TensorTemplate<TZeroPointA>* a_zp;
    TosaReference::TensorTemplate<TZeroPointB>* b_zp;
    TosaReference::TensorTemplate<TOut>* output;
    int64_t N;
    int64_t H;
    int64_t W;
    int64_t C;
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
    std::unique_ptr<tosa::TosaMaxPool2dAttribute> attribute;
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
    std::unique_ptr<tosa::TosaFFT2dAttribute> attribute;
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
    std::unique_ptr<tosa::TosaRFFT2dAttribute> attribute;
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
    using TInZp           = Eigen::Tensor<InEigenType, 1>;
    using TWeightZp       = Eigen::Tensor<WeightEigenType, 1>;
    using TOut            = Eigen::Tensor<OutEigenType, 4>;
    using TAcc            = Eigen::Tensor<AccEigenType, 4>;

    static constexpr int64_t AccQMin = GetQMin<OutDtype>::value;
    static constexpr int64_t AccQMax = GetQMax<OutDtype>::value;

protected:
    TosaReference::TensorTemplate<TIn>* input;
    TosaReference::TensorTemplate<TWeight>* weight;
    TosaReference::TensorTemplate<TBias>* bias;
    TosaReference::TensorTemplate<TOut>* output;
    std::unique_ptr<TosaTransposeConv2dAttribute> attribute;
};

};    // namespace TosaReference

#endif

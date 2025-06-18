// Copyright (c) 2025 ARM Limited.
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

#ifndef ADVANCED_TENSOR_OPS_H
#define ADVANCED_TENSOR_OPS_H

#include "graph_node.h"
#include "tensor.h"
#include <complex>

using namespace tosa;

namespace TosaReference
{

// Advanced tensor fusion operation with multiple strategies
template <int Rank, TOSA_REF_TYPE Dtype>
class OpTensorFusion : public GraphNode
{
public:
    OpTensorFusion(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_);
    virtual ~OpTensorFusion();
    virtual int eval() override;
    static constexpr int InRank  = Rank;
    static constexpr int OutRank = Rank;
    static constexpr TOSA_REF_TYPE InDtype  = Dtype;
    static constexpr TOSA_REF_TYPE OutDtype = Dtype;

protected:
    TosaFusionAttribute* attribute;
    Tensor* input1;
    Tensor* input2;
    Tensor* input3;  // Optional third input
    Tensor* output;
    
    // Fusion strategies
    enum FusionStrategy {
        CONCATENATE = 0,
        ELEMENT_WISE_ADD = 1,
        WEIGHTED_SUM = 2,
        ATTENTION_FUSION = 3
    };
};

// Spectral transform operations (FFT/IFFT)
template <int Rank, TOSA_REF_TYPE Dtype>
class OpSpectralTransform : public GraphNode
{
public:
    OpSpectralTransform(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_);
    virtual ~OpSpectralTransform();
    virtual int eval() override;
    static constexpr int InRank  = Rank;
    static constexpr int OutRank = Rank;
    static constexpr TOSA_REF_TYPE InDtype  = Dtype;
    static constexpr TOSA_REF_TYPE OutDtype = Dtype;

protected:
    TosaSpectralAttribute* attribute;
    Tensor* input;
    Tensor* output;
    
    // Transform types
    enum TransformType {
        FFT = 0,
        IFFT = 1,
        REAL_FFT = 2,
        DCT = 3
    };
    
    // Helper functions for FFT implementation
    void computeFFT(const std::complex<float>* input, std::complex<float>* output, int n, bool inverse);
    void cooleyTukeyFFT(std::complex<float>* data, int n, bool inverse);
};

// Advanced activation functions beyond standard TOSA
template <int Rank, TOSA_REF_TYPE Dtype>
class OpAdvancedActivation : public GraphNode
{
public:
    OpAdvancedActivation(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_);
    virtual ~OpAdvancedActivation();
    virtual int eval() override;
    static constexpr int InRank  = Rank;
    static constexpr int OutRank = Rank;
    static constexpr TOSA_REF_TYPE InDtype  = Dtype;
    static constexpr TOSA_REF_TYPE OutDtype = Dtype;

protected:
    TosaActivationAttribute* attribute;
    Tensor* input;
    Tensor* output;
    
    // Advanced activation types
    enum ActivationType {
        MISH = 0,
        SWISH = 1,
        GELU_TANH = 2,
        GELU_ERF = 3,
        SELU = 4,
        ELU = 5,
        HARDSWISH = 6
    };
    
    using InEigenType  = typename GetEigenType<Dtype>::type;
    using OutEigenType = typename GetEigenType<Dtype>::type;
    using TIn  = Eigen::Tensor<InEigenType, InRank>;
    using TOut = Eigen::Tensor<OutEigenType, OutRank>;
};

// Tensor decomposition operations (SVD, QR, etc.)
template <int Rank, TOSA_REF_TYPE Dtype>
class OpTensorDecomposition : public GraphNode
{
public:
    OpTensorDecomposition(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_);
    virtual ~OpTensorDecomposition();
    virtual int eval() override;
    static constexpr int InRank  = Rank;
    static constexpr int OutRank = Rank;
    static constexpr TOSA_REF_TYPE InDtype  = Dtype;
    static constexpr TOSA_REF_TYPE OutDtype = Dtype;

protected:
    TosaDecompositionAttribute* attribute;
    Tensor* input;
    Tensor* output_u;  // For SVD: U matrix
    Tensor* output_s;  // For SVD: singular values
    Tensor* output_v;  // For SVD: V matrix
    
    enum DecompositionType {
        SVD = 0,
        QR = 1,
        LU = 2,
        CHOLESKY = 3
    };
};

// Statistical operations for tensor analysis
template <int Rank, TOSA_REF_TYPE Dtype>
class OpStatisticalOps : public GraphNode
{
public:
    OpStatisticalOps(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_);
    virtual ~OpStatisticalOps();
    virtual int eval() override;
    static constexpr int InRank  = Rank;
    static constexpr int OutRank = Rank;
    static constexpr TOSA_REF_TYPE InDtype  = Dtype;
    static constexpr TOSA_REF_TYPE OutDtype = Dtype;

protected:
    TosaStatisticalAttribute* attribute;
    Tensor* input;
    Tensor* input2;  // For operations requiring two inputs
    Tensor* output;
    
    enum StatisticalType {
        ENTROPY = 0,
        MUTUAL_INFORMATION = 1,
        CORRELATION = 2,
        COVARIANCE = 3,
        MOMENT = 4,
        SKEWNESS = 5,
        KURTOSIS = 6
    };
};

// Geometric operations and distance metrics
template <int Rank, TOSA_REF_TYPE Dtype>
class OpGeometricOps : public GraphNode
{
public:
    OpGeometricOps(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_);
    virtual ~OpGeometricOps();
    virtual int eval() override;
    static constexpr int InRank  = Rank;
    static constexpr int OutRank = Rank;
    static constexpr TOSA_REF_TYPE InDtype  = Dtype;
    static constexpr TOSA_REF_TYPE OutDtype = Dtype;

protected:
    TosaGeometricAttribute* attribute;
    Tensor* input1;
    Tensor* input2;
    Tensor* output;
    
    enum GeometricType {
        EUCLIDEAN_DISTANCE = 0,
        MANHATTAN_DISTANCE = 1,
        COSINE_SIMILARITY = 2,
        DOT_PRODUCT = 3,
        CROSS_PRODUCT = 4,
        AFFINE_TRANSFORM = 5,
        PERSPECTIVE_TRANSFORM = 6
    };
};

}    // namespace TosaReference

#endif
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

#include "advanced_tensor_ops.h"
#include "tosa_advanced_attributes.h"
#include "arith_util.h"
#include "template_types.h"
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/QR>
#include <cmath>
#include <algorithm>
#include <numeric>

using namespace tosa;

namespace TosaReference
{

// TensorDecomposition Implementation
template <int Rank, TOSA_REF_TYPE Dtype>
OpTensorDecomposition<Rank, Dtype>::OpTensorDecomposition(SubgraphTraverser* sgt_,
                                                           TosaAttributeBase* attribute_,
                                                           uint64_t id_)
    : GraphNode(sgt_, Op_CUSTOM, id_)
{
    setRequiredOperands(1, 3);  // 1 input, up to 3 outputs (U, S, V)
    setRequiredRank(Rank);
    
    INIT_ATTRIBUTE(Decomposition);
}

template <int Rank, TOSA_REF_TYPE Dtype>
OpTensorDecomposition<Rank, Dtype>::~OpTensorDecomposition()
{
    if (attribute)
        delete attribute;
}

template <int Rank, TOSA_REF_TYPE Dtype>
int OpTensorDecomposition<Rank, Dtype>::eval()
{
    input = dynamic_cast<TensorTemplate<TIn>*>(inputs[0]);
    output_u = dynamic_cast<TensorTemplate<TOut>*>(outputs[0]);
    output_s = dynamic_cast<TensorTemplate<TOut>*>(outputs[1]);
    output_v = dynamic_cast<TensorTemplate<TOut>*>(outputs[2]);
    
    ASSERT_MEM(input && output_u && output_s && output_v);
    
    DecompositionType decomp_type = static_cast<DecompositionType>(attribute->decomposition_type());
    
    // Get input matrix (assume last two dimensions are the matrix)
    auto input_shape = input->getShape();
    int m = input_shape[input_shape.size() - 2];
    int n = input_shape[input_shape.size() - 1];
    
    // Convert to Eigen matrix
    Eigen::MatrixXf matrix(m, n);
    auto input_tensor = input->getTensor();
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            matrix(i, j) = input_tensor(i * n + j);
        }
    }
    
    switch (decomp_type) {
        case SVD: {
            // Singular Value Decomposition
            Eigen::JacobiSVD<Eigen::MatrixXf> svd(matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
            
            auto U = svd.matrixU();
            auto S = svd.singularValues();
            auto V = svd.matrixV();
            
            // Copy results to output tensors
            auto& u_tensor = output_u->getTensor();
            auto& s_tensor = output_s->getTensor();
            auto& v_tensor = output_v->getTensor();
            
            for (int i = 0; i < U.rows(); i++) {
                for (int j = 0; j < U.cols(); j++) {
                    u_tensor(i * U.cols() + j) = U(i, j);
                }
            }
            
            for (int i = 0; i < S.size(); i++) {
                s_tensor(i) = S(i);
            }
            
            for (int i = 0; i < V.rows(); i++) {
                for (int j = 0; j < V.cols(); j++) {
                    v_tensor(i * V.cols() + j) = V(i, j);
                }
            }
            break;
        }
        
        case QR: {
            // QR Decomposition
            Eigen::HouseholderQR<Eigen::MatrixXf> qr(matrix);
            auto Q = qr.householderQ() * Eigen::MatrixXf::Identity(m, std::min(m, n));
            auto R = qr.matrixQR().triangularView<Eigen::Upper>();
            
            // Copy Q to output_u and R to output_s (reusing outputs)
            auto& q_tensor = output_u->getTensor();
            auto& r_tensor = output_s->getTensor();
            
            for (int i = 0; i < Q.rows(); i++) {
                for (int j = 0; j < Q.cols(); j++) {
                    q_tensor(i * Q.cols() + j) = Q(i, j);
                }
            }
            
            for (int i = 0; i < std::min(m, n); i++) {
                for (int j = i; j < n; j++) {
                    r_tensor(i * n + j) = R(i, j);
                }
            }
            break;
        }
        
        case LU: {
            // LU Decomposition with partial pivoting
            Eigen::PartialPivLU<Eigen::MatrixXf> lu(matrix);
            auto L = Eigen::MatrixXf::Identity(m, m);
            L.triangularView<Eigen::StrictlyLower>() = lu.matrixLU();
            auto U = lu.matrixLU().triangularView<Eigen::Upper>();
            
            // Copy L and U to outputs
            auto& l_tensor = output_u->getTensor();
            auto& u_tensor = output_s->getTensor();
            
            for (int i = 0; i < L.rows(); i++) {
                for (int j = 0; j < L.cols(); j++) {
                    l_tensor(i * L.cols() + j) = L(i, j);
                }
            }
            
            for (int i = 0; i < U.rows(); i++) {
                for (int j = 0; j < U.cols(); j++) {
                    u_tensor(i * U.cols() + j) = U(i, j);
                }
            }
            break;
        }
        
        case CHOLESKY: {
            // Cholesky Decomposition (for positive definite matrices)
            Eigen::LLT<Eigen::MatrixXf> chol(matrix);
            if (chol.info() != Eigen::Success) {
                FATAL_ERROR("OpTensorDecomposition: Cholesky decomposition failed - matrix not positive definite");
            }
            
            auto L = chol.matrixL();
            auto& l_tensor = output_u->getTensor();
            
            for (int i = 0; i < L.rows(); i++) {
                for (int j = 0; j < L.cols(); j++) {
                    l_tensor(i * L.cols() + j) = L(i, j);
                }
            }
            break;
        }
        
        default:
            FATAL_ERROR("OpTensorDecomposition: unsupported decomposition type");
    }
    
    return GraphNode::eval();
}

// StatisticalOps Implementation
template <int Rank, TOSA_REF_TYPE Dtype>
OpStatisticalOps<Rank, Dtype>::OpStatisticalOps(SubgraphTraverser* sgt_,
                                                 TosaAttributeBase* attribute_,
                                                 uint64_t id_)
    : GraphNode(sgt_, Op_CUSTOM, id_)
{
    setRequiredOperands(1, 1);  // May have 2 inputs for some operations
    setRequiredRank(Rank);
    
    INIT_ATTRIBUTE(Statistical);
}

template <int Rank, TOSA_REF_TYPE Dtype>
OpStatisticalOps<Rank, Dtype>::~OpStatisticalOps()
{
    if (attribute)
        delete attribute;
}

template <int Rank, TOSA_REF_TYPE Dtype>
int OpStatisticalOps<Rank, Dtype>::eval()
{
    input = dynamic_cast<TensorTemplate<TIn>*>(inputs[0]);
    output = dynamic_cast<TensorTemplate<TOut>*>(outputs[0]);
    
    if (inputs.size() > 1) {
        input2 = dynamic_cast<TensorTemplate<TIn>*>(inputs[1]);
    }
    
    ASSERT_MEM(input && output);
    
    StatisticalType stat_type = static_cast<StatisticalType>(attribute->statistical_type());
    
    switch (stat_type) {
        case ENTROPY: {
            // Shannon entropy: H(X) = -∑ p(x) log p(x)
            auto input_tensor = input->getTensor();
            auto& output_tensor = output->getTensor();
            
            // Normalize to get probabilities
            float sum = input_tensor.sum();
            auto probabilities = input_tensor / sum;
            
            // Compute entropy
            auto log_probs = probabilities.unaryExpr([](float p) -> float {
                return p > 0 ? p * std::log(p) : 0.0f;
            });
            
            float entropy = -log_probs.sum();
            output_tensor.setConstant(entropy);
            break;
        }
        
        case MUTUAL_INFORMATION: {
            // Mutual information between two variables
            ASSERT_MEM(input2);
            
            auto x_tensor = input->getTensor();
            auto y_tensor = input2->getTensor();
            auto& output_tensor = output->getTensor();
            
            // Simplified mutual information calculation
            // This is a basic implementation - real MI requires proper binning
            float mean_x = x_tensor.mean();
            float mean_y = y_tensor.mean();
            float var_x = ((x_tensor - mean_x).square()).mean();
            float var_y = ((y_tensor - mean_y).square()).mean();
            float cov_xy = ((x_tensor - mean_x) * (y_tensor - mean_y)).mean();
            
            float correlation = cov_xy / std::sqrt(var_x * var_y);
            float mi = -0.5f * std::log(1.0f - correlation * correlation);
            
            output_tensor.setConstant(mi);
            break;
        }
        
        case CORRELATION: {
            // Pearson correlation coefficient
            ASSERT_MEM(input2);
            
            auto x_tensor = input->getTensor();
            auto y_tensor = input2->getTensor();
            auto& output_tensor = output->getTensor();
            
            float mean_x = x_tensor.mean();
            float mean_y = y_tensor.mean();
            
            auto numerator = ((x_tensor - mean_x) * (y_tensor - mean_y)).sum();
            auto denom_x = ((x_tensor - mean_x).square()).sum();
            auto denom_y = ((y_tensor - mean_y).square()).sum();
            
            float correlation = numerator / std::sqrt(denom_x * denom_y);
            output_tensor.setConstant(correlation);
            break;
        }
        
        case MOMENT: {
            // Central moment of specified order
            auto input_tensor = input->getTensor();
            auto& output_tensor = output->getTensor();
            
            int order = attribute->moment_order();
            float mean = input_tensor.mean();
            
            auto moment_tensor = (input_tensor - mean).pow(order);
            float moment = moment_tensor.mean();
            
            output_tensor.setConstant(moment);
            break;
        }
        
        case SKEWNESS: {
            // Third standardized moment
            auto input_tensor = input->getTensor();
            auto& output_tensor = output->getTensor();
            
            float mean = input_tensor.mean();
            float variance = ((input_tensor - mean).square()).mean();
            float std_dev = std::sqrt(variance);
            
            auto skew_tensor = ((input_tensor - mean) / std_dev).pow(3);
            float skewness = skew_tensor.mean();
            
            output_tensor.setConstant(skewness);
            break;
        }
        
        case KURTOSIS: {
            // Fourth standardized moment
            auto input_tensor = input->getTensor();
            auto& output_tensor = output->getTensor();
            
            float mean = input_tensor.mean();
            float variance = ((input_tensor - mean).square()).mean();
            float std_dev = std::sqrt(variance);
            
            auto kurt_tensor = ((input_tensor - mean) / std_dev).pow(4);
            float kurtosis = kurt_tensor.mean() - 3.0f;  // Excess kurtosis
            
            output_tensor.setConstant(kurtosis);
            break;
        }
        
        default:
            FATAL_ERROR("OpStatisticalOps: unsupported statistical operation type");
    }
    
    return GraphNode::eval();
}

// GeometricOps Implementation
template <int Rank, TOSA_REF_TYPE Dtype>
OpGeometricOps<Rank, Dtype>::OpGeometricOps(SubgraphTraverser* sgt_,
                                             TosaAttributeBase* attribute_,
                                             uint64_t id_)
    : GraphNode(sgt_, Op_CUSTOM, id_)
{
    setRequiredOperands(2, 1);
    setRequiredRank(Rank);
    
    INIT_ATTRIBUTE(Geometric);
}

template <int Rank, TOSA_REF_TYPE Dtype>
OpGeometricOps<Rank, Dtype>::~OpGeometricOps()
{
    if (attribute)
        delete attribute;
}

template <int Rank, TOSA_REF_TYPE Dtype>
int OpGeometricOps<Rank, Dtype>::eval()
{
    input1 = dynamic_cast<TensorTemplate<TIn>*>(inputs[0]);
    input2 = dynamic_cast<TensorTemplate<TIn>*>(inputs[1]);
    output = dynamic_cast<TensorTemplate<TOut>*>(outputs[0]);
    
    ASSERT_MEM(input1 && input2 && output);
    
    GeometricType geom_type = static_cast<GeometricType>(attribute->geometric_type());
    
    switch (geom_type) {
        case EUCLIDEAN_DISTANCE: {
            // L2 distance between vectors
            auto diff = input1->getTensor() - input2->getTensor();
            auto distance = diff.square().sum().sqrt();
            output->getTensor().setConstant(distance);
            break;
        }
        
        case MANHATTAN_DISTANCE: {
            // L1 distance between vectors
            auto diff = input1->getTensor() - input2->getTensor();
            auto distance = diff.abs().sum();
            output->getTensor().setConstant(distance);
            break;
        }
        
        case COSINE_SIMILARITY: {
            // Cosine similarity between vectors
            auto dot_product = (input1->getTensor() * input2->getTensor()).sum();
            auto norm1 = input1->getTensor().square().sum().sqrt();
            auto norm2 = input2->getTensor().square().sum().sqrt();
            auto similarity = dot_product / (norm1 * norm2);
            output->getTensor().setConstant(similarity);
            break;
        }
        
        case DOT_PRODUCT: {
            // Dot product between vectors
            auto dot = (input1->getTensor() * input2->getTensor()).sum();
            output->getTensor().setConstant(dot);
            break;
        }
        
        case CROSS_PRODUCT: {
            // Cross product for 3D vectors
            auto a = input1->getTensor();
            auto b = input2->getTensor();
            auto& c = output->getTensor();
            
            // Assuming 3D vectors [x, y, z]
            c(0) = a(1) * b(2) - a(2) * b(1);
            c(1) = a(2) * b(0) - a(0) * b(2);
            c(2) = a(0) * b(1) - a(1) * b(0);
            break;
        }
        
        case AFFINE_TRANSFORM: {
            // Apply affine transformation matrix
            const auto& transform_matrix = attribute->transform_matrix();
            
            // Apply transformation (simplified for vectors)
            auto input_vec = input1->getTensor();
            auto& output_vec = output->getTensor();
            
            // Assuming 2D transformation: [x', y'] = M * [x, y] + t
            if (transform_matrix.size() >= 6) {
                float m11 = transform_matrix[0], m12 = transform_matrix[1];
                float m21 = transform_matrix[2], m22 = transform_matrix[3];
                float tx = transform_matrix[4], ty = transform_matrix[5];
                
                output_vec(0) = m11 * input_vec(0) + m12 * input_vec(1) + tx;
                output_vec(1) = m21 * input_vec(0) + m22 * input_vec(1) + ty;
            }
            break;
        }
        
        default:
            FATAL_ERROR("OpGeometricOps: unsupported geometric operation type");
    }
    
    return GraphNode::eval();
}

// Explicit template instantiations for remaining operations
DEF_INSTANTIATE_RANK0_6_ONE_TYPE(OpTensorDecomposition, FP32);
DEF_INSTANTIATE_RANK0_6_ONE_TYPE(OpTensorDecomposition, FP64);

DEF_INSTANTIATE_RANK0_6_ONE_TYPE(OpStatisticalOps, FP32);
DEF_INSTANTIATE_RANK0_6_ONE_TYPE(OpStatisticalOps, FP64);

DEF_INSTANTIATE_RANK0_6_ONE_TYPE(OpGeometricOps, FP16);
DEF_INSTANTIATE_RANK0_6_ONE_TYPE(OpGeometricOps, FP32);
DEF_INSTANTIATE_RANK0_6_ONE_TYPE(OpGeometricOps, FP64);

}    // namespace TosaReference
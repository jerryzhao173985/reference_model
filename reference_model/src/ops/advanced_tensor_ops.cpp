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
#include "arith_util.h"
#include "quant_util.h"
#include "template_types.h"
#include <cmath>
#include <algorithm>
#include <numeric>

using namespace tosa;

namespace TosaReference
{

// TensorFusion Implementation
template <int Rank, TOSA_REF_TYPE Dtype>
OpTensorFusion<Rank, Dtype>::OpTensorFusion(SubgraphTraverser* sgt_,
                                             TosaAttributeBase* attribute_,
                                             uint64_t id_)
    : GraphNode(sgt_, Op_CUSTOM, id_)
{
    setRequiredOperands(2, 1);  // Minimum 2 inputs, 1 output
    setRequiredRank(Rank);
    
    INIT_ATTRIBUTE(Fusion);
}

template <int Rank, TOSA_REF_TYPE Dtype>
OpTensorFusion<Rank, Dtype>::~OpTensorFusion()
{
    if (attribute)
        delete attribute;
}

template <int Rank, TOSA_REF_TYPE Dtype>
int OpTensorFusion<Rank, Dtype>::eval()
{
    // Get input and output tensors
    input1 = dynamic_cast<TensorTemplate<TIn>*>(inputs[0]);
    input2 = dynamic_cast<TensorTemplate<TIn>*>(inputs[1]);
    output = dynamic_cast<TensorTemplate<TOut>*>(outputs[0]);
    
    if (inputs.size() > 2) {
        input3 = dynamic_cast<TensorTemplate<TIn>*>(inputs[2]);
    }
    
    ASSERT_MEM(input1 && input2 && output);
    
    // Perform fusion based on strategy
    FusionStrategy strategy = static_cast<FusionStrategy>(attribute->strategy());
    
    switch (strategy) {
        case CONCATENATE: {
            // Concatenate along specified axis
            int axis = attribute->axis();
            auto concatenated = input1->getTensor().concatenate(input2->getTensor(), axis);
            output->getTensor() = concatenated;
            break;
        }
        
        case ELEMENT_WISE_ADD: {
            // Element-wise addition with broadcasting
            output->getTensor() = input1->getTensor() + input2->getTensor();
            break;
        }
        
        case WEIGHTED_SUM: {
            // Weighted sum with learnable weights
            float weight1 = attribute->weight1();
            float weight2 = attribute->weight2();
            output->getTensor() = weight1 * input1->getTensor() + weight2 * input2->getTensor();
            break;
        }
        
        case ATTENTION_FUSION: {
            // Attention-based fusion mechanism
            auto attention_weights = input1->getTensor().square().sum() / 
                                   (input1->getTensor().square().sum() + input2->getTensor().square().sum());
            output->getTensor() = attention_weights * input1->getTensor() + 
                                (1.0f - attention_weights) * input2->getTensor();
            break;
        }
        
        default:
            FATAL_ERROR("OpTensorFusion: unsupported fusion strategy");
    }
    
    return GraphNode::eval();
}

// SpectralTransform Implementation
template <int Rank, TOSA_REF_TYPE Dtype>
OpSpectralTransform<Rank, Dtype>::OpSpectralTransform(SubgraphTraverser* sgt_,
                                                       TosaAttributeBase* attribute_,
                                                       uint64_t id_)
    : GraphNode(sgt_, Op_CUSTOM, id_)
{
    setRequiredOperands(1, 1);
    setRequiredRank(Rank);
    
    INIT_ATTRIBUTE(Spectral);
}

template <int Rank, TOSA_REF_TYPE Dtype>
OpSpectralTransform<Rank, Dtype>::~OpSpectralTransform()
{
    if (attribute)
        delete attribute;
}

template <int Rank, TOSA_REF_TYPE Dtype>
int OpSpectralTransform<Rank, Dtype>::eval()
{
    input = dynamic_cast<TensorTemplate<TIn>*>(inputs[0]);
    output = dynamic_cast<TensorTemplate<TOut>*>(outputs[0]);
    
    ASSERT_MEM(input && output);
    
    TransformType transform_type = static_cast<TransformType>(attribute->transform_type());
    
    switch (transform_type) {
        case FFT:
        case IFFT: {
            // Implement FFT using Cooley-Tukey algorithm
            auto input_shape = input->getShape();
            int n = input_shape[input_shape.size() - 1];  // Transform along last dimension
            
            // Convert input to complex
            std::vector<std::complex<float>> complex_data(n);
            auto input_tensor = input->getTensor();
            
            // For simplicity, process first batch/channel
            for (int i = 0; i < n; i++) {
                complex_data[i] = std::complex<float>(input_tensor(i), 0.0f);
            }
            
            // Perform FFT
            std::vector<std::complex<float>> output_data(n);
            computeFFT(complex_data.data(), output_data.data(), n, transform_type == IFFT);
            
            // Convert back to real (magnitude)
            auto& output_tensor = output->getTensor();
            for (int i = 0; i < n; i++) {
                output_tensor(i) = std::abs(output_data[i]);
            }
            break;
        }
        
        case REAL_FFT: {
            // Real-valued FFT optimization
            // Implementation details...
            break;
        }
        
        case DCT: {
            // Discrete Cosine Transform
            auto input_tensor = input->getTensor();
            auto& output_tensor = output->getTensor();
            
            int n = input->getShape()[input->getShape().size() - 1];
            for (int k = 0; k < n; k++) {
                float sum = 0.0f;
                for (int i = 0; i < n; i++) {
                    sum += input_tensor(i) * std::cos(M_PI * k * (2*i + 1) / (2*n));
                }
                output_tensor(k) = sum * std::sqrt(2.0f / n) * (k == 0 ? 1.0f / std::sqrt(2.0f) : 1.0f);
            }
            break;
        }
    }
    
    return GraphNode::eval();
}

template <int Rank, TOSA_REF_TYPE Dtype>
void OpSpectralTransform<Rank, Dtype>::computeFFT(const std::complex<float>* input,
                                                   std::complex<float>* output,
                                                   int n, bool inverse)
{
    // Copy input to output for in-place processing
    std::copy(input, input + n, output);
    cooleyTukeyFFT(output, n, inverse);
}

template <int Rank, TOSA_REF_TYPE Dtype>
void OpSpectralTransform<Rank, Dtype>::cooleyTukeyFFT(std::complex<float>* data, int n, bool inverse)
{
    if (n <= 1) return;
    
    // Divide
    std::vector<std::complex<float>> even(n/2), odd(n/2);
    for (int i = 0; i < n/2; i++) {
        even[i] = data[2*i];
        odd[i] = data[2*i + 1];
    }
    
    // Conquer
    cooleyTukeyFFT(even.data(), n/2, inverse);
    cooleyTukeyFFT(odd.data(), n/2, inverse);
    
    // Combine
    for (int i = 0; i < n/2; i++) {
        float angle = (inverse ? 2 : -2) * M_PI * i / n;
        std::complex<float> t = std::polar(1.0f, angle) * odd[i];
        data[i] = even[i] + t;
        data[i + n/2] = even[i] - t;
    }
    
    if (inverse) {
        for (int i = 0; i < n; i++) {
            data[i] /= 2;
        }
    }
}

// AdvancedActivation Implementation
template <int Rank, TOSA_REF_TYPE Dtype>
OpAdvancedActivation<Rank, Dtype>::OpAdvancedActivation(SubgraphTraverser* sgt_,
                                                         TosaAttributeBase* attribute_,
                                                         uint64_t id_)
    : GraphNode(sgt_, Op_CUSTOM, id_)
{
    setRequiredOperands(1, 1);
    setRequiredRank(Rank);
    
    INIT_ATTRIBUTE(Activation);
}

template <int Rank, TOSA_REF_TYPE Dtype>
OpAdvancedActivation<Rank, Dtype>::~OpAdvancedActivation()
{
    if (attribute)
        delete attribute;
}

template <int Rank, TOSA_REF_TYPE Dtype>
int OpAdvancedActivation<Rank, Dtype>::eval()
{
    input = dynamic_cast<TensorTemplate<TIn>*>(inputs[0]);
    output = dynamic_cast<TensorTemplate<TOut>*>(outputs[0]);
    
    ASSERT_MEM(input && output);
    
    ActivationType activation_type = static_cast<ActivationType>(attribute->activation_type());
    
    switch (activation_type) {
        case MISH: {
            // Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))
            auto softplus = [](InEigenType x) -> OutEigenType {
                return std::log(1.0f + std::exp(x));
            };
            auto mish = [&softplus](InEigenType x) -> OutEigenType {
                return x * std::tanh(softplus(x));
            };
            output->getTensor() = input->getTensor().unaryExpr(mish);
            break;
        }
        
        case SWISH: {
            // Swish(x) = x * sigmoid(x)
            auto swish = [](InEigenType x) -> OutEigenType {
                return x / (1.0f + std::exp(-x));
            };
            output->getTensor() = input->getTensor().unaryExpr(swish);
            break;
        }
        
        case GELU_TANH: {
            // GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
            auto gelu_tanh = [](InEigenType x) -> OutEigenType {
                constexpr float sqrt_2_over_pi = 0.7978845608f;
                float tanh_arg = sqrt_2_over_pi * (x + 0.044715f * x * x * x);
                return 0.5f * x * (1.0f + std::tanh(tanh_arg));
            };
            output->getTensor() = input->getTensor().unaryExpr(gelu_tanh);
            break;
        }
        
        case GELU_ERF: {
            // GELU(x) = 0.5 * x * (1 + erf(x / √2))
            auto gelu_erf = [](InEigenType x) -> OutEigenType {
                return 0.5f * x * (1.0f + std::erf(x / std::sqrt(2.0f)));
            };
            output->getTensor() = input->getTensor().unaryExpr(gelu_erf);
            break;
        }
        
        case SELU: {
            // SELU(x) = λ * (α * (e^x - 1) if x < 0 else x)
            constexpr float alpha = 1.6732632423543772848170429916717f;
            constexpr float lambda = 1.0507009873554804934193349852946f;
            auto selu = [](InEigenType x) -> OutEigenType {
                return x >= 0 ? lambda * x : lambda * alpha * (std::exp(x) - 1.0f);
            };
            output->getTensor() = input->getTensor().unaryExpr(selu);
            break;
        }
        
        case ELU: {
            // ELU(x) = α * (e^x - 1) if x < 0 else x
            float alpha = attribute->alpha();
            auto elu = [alpha](InEigenType x) -> OutEigenType {
                return x >= 0 ? x : alpha * (std::exp(x) - 1.0f);
            };
            output->getTensor() = input->getTensor().unaryExpr(elu);
            break;
        }
        
        case HARDSWISH: {
            // HardSwish(x) = x * ReLU6(x + 3) / 6
            auto hardswish = [](InEigenType x) -> OutEigenType {
                float relu6 = std::min(6.0f, std::max(0.0f, x + 3.0f));
                return x * relu6 / 6.0f;
            };
            output->getTensor() = input->getTensor().unaryExpr(hardswish);
            break;
        }
        
        default:
            FATAL_ERROR("OpAdvancedActivation: unsupported activation type");
    }
    
    return GraphNode::eval();
}

// Explicit template instantiations
DEF_INSTANTIATE_RANK0_6_ONE_TYPE(OpTensorFusion, FP16);
DEF_INSTANTIATE_RANK0_6_ONE_TYPE(OpTensorFusion, FP32);
DEF_INSTANTIATE_RANK0_6_ONE_TYPE(OpTensorFusion, FP64);

DEF_INSTANTIATE_RANK0_6_ONE_TYPE(OpSpectralTransform, FP32);
DEF_INSTANTIATE_RANK0_6_ONE_TYPE(OpSpectralTransform, FP64);

DEF_INSTANTIATE_RANK0_6_ONE_TYPE(OpAdvancedActivation, FP16);
DEF_INSTANTIATE_RANK0_6_ONE_TYPE(OpAdvancedActivation, FP32);
DEF_INSTANTIATE_RANK0_6_ONE_TYPE(OpAdvancedActivation, FP64);

}    // namespace TosaReference
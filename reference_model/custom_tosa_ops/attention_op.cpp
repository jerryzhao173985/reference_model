// Copyright (c) 2025, ARM Limited.
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

#include "custom_op_interface.h"
#include "custom_registry.h"
#include <json.hpp>
#include <vector>
#include <cmath>

#ifdef _MSC_VER
#define TOSA_EXPORT __declspec(dllexport)
#else
#define TOSA_EXPORT
#endif

using namespace tosa;
using json = nlohmann::json;

namespace TosaReference
{

/**
 * @brief Custom Attention Operation
 * 
 * This operation implements various attention mechanisms used in transformer models.
 * It supports scaled dot-product attention as used in the original transformer paper,
 * as well as more efficient attention variants.
 *
 * Inputs:
 * - query: Tensor of shape [batch_size, num_heads, seq_len_q, head_size]
 * - key: Tensor of shape [batch_size, num_heads, seq_len_k, head_size]
 * - value: Tensor of shape [batch_size, num_heads, seq_len_k, head_size]
 * - mask (optional): Tensor of shape [batch_size, num_heads, seq_len_q, seq_len_k]
 *
 * Output:
 * - attention_output: Tensor of shape [batch_size, num_heads, seq_len_q, head_size]
 */
class AttentionOp : public CustomOpInterface
{
public:
    AttentionOp() = default;
    AttentionOp(std::string& domain_name, std::string& operator_name, std::string& version)
        : _domain_name(domain_name)
        , _operator_name(operator_name)
        , _version(version)
    {}

    int eval(std::vector<TosaReference::Tensor*>& input_tensors,
             std::vector<TosaReference::Tensor*>& output_tensors,
             const std::string& implementation_attrs) override
    {
        // Parse the implementation attributes as JSON
        json attrs;
        try {
            attrs = json::parse(implementation_attrs);
        } catch (const json::parse_error& e) {
            std::cerr << "Error parsing implementation_attrs JSON: " << e.what() << std::endl;
            return 1;
        }

        // Get the attention type from attributes
        std::string attention_type = attrs.value("attention_type", "scaled_dot_product");
        float scale = attrs.value("scale", 0.0f); // Scale factor for attention scores

        // Validate input tensors
        if (input_tensors.size() < 3 || output_tensors.empty()) {
            std::cerr << "Attention requires at least 3 input tensors (Q, K, V) and 1 output tensor" << std::endl;
            return 1;
        }

        // Get query, key, value tensors
        auto query_tensor = input_tensors[0];
        auto key_tensor = input_tensors[1];
        auto value_tensor = input_tensors[2];
        auto output_tensor = output_tensors[0];

        // Get mask tensor if provided
        TosaReference::Tensor* mask_tensor = nullptr;
        if (input_tensors.size() > 3) {
            mask_tensor = input_tensors[3];
        }

        // Only float types supported for now
        if (query_tensor->getDType() != DType::FLOAT32 ||
            key_tensor->getDType() != DType::FLOAT32 ||
            value_tensor->getDType() != DType::FLOAT32) {
            std::cerr << "Attention currently only supports FLOAT32 tensors" << std::endl;
            return 1;
        }

        // Basic validation of tensor shapes
        auto query_shape = query_tensor->getShape();
        auto key_shape = key_tensor->getShape();
        auto value_shape = value_tensor->getShape();
        auto output_shape = output_tensor->getShape();

        // Ensure tensors have rank 4
        if (query_shape.size() != 4 || key_shape.size() != 4 || 
            value_shape.size() != 4 || output_shape.size() != 4) {
            std::cerr << "Attention requires tensors with rank 4" << std::endl;
            return 1;
        }

        // Ensure batch_size and num_heads dimensions match
        if (query_shape[0] != key_shape[0] || query_shape[0] != value_shape[0] ||
            query_shape[1] != key_shape[1] || query_shape[1] != value_shape[1]) {
            std::cerr << "Batch size and number of heads must match for Q, K, V tensors" << std::endl;
            return 1;
        }

        // Ensure key and value sequence length match
        if (key_shape[2] != value_shape[2]) {
            std::cerr << "Key and value sequence lengths must match" << std::endl;
            return 1;
        }

        // Ensure head dimensions match
        if (query_shape[3] != key_shape[3]) {
            std::cerr << "Query and key head dimensions must match" << std::endl;
            return 1;
        }

        // Ensure output shape matches expected dimensions
        if (output_shape[0] != query_shape[0] || output_shape[1] != query_shape[1] ||
            output_shape[2] != query_shape[2] || output_shape[3] != value_shape[3]) {
            std::cerr << "Output tensor shape does not match expected dimensions" << std::endl;
            return 1;
        }

        // Get dimensions
        int batch_size = query_shape[0];
        int num_heads = query_shape[1];
        int seq_len_q = query_shape[2];
        int seq_len_k = key_shape[2];
        int head_size = query_shape[3];

        // Calculate scale factor if not provided
        if (scale <= 0.0f) {
            scale = 1.0f / std::sqrt(static_cast<float>(head_size));
        }

        // For each attention type, delegate to the appropriate function
        if (attention_type == "scaled_dot_product") {
            return evaluateScaledDotProductAttention(query_tensor, key_tensor, value_tensor, 
                                                      mask_tensor, output_tensor, scale);
        } else {
            std::cerr << "Unknown attention type: " << attention_type << std::endl;
            return 1;
        }
    }

    std::string getDomainName() const override
    {
        return this->_domain_name;
    }

    std::string getOperatorName() const override
    {
        return this->_operator_name;
    }

    std::string getVersion() const override
    {
        return this->_version;
    }

    ~AttentionOp(){}

private:
    int evaluateScaledDotProductAttention(TosaReference::Tensor* query_tensor,
                                          TosaReference::Tensor* key_tensor,
                                          TosaReference::Tensor* value_tensor,
                                          TosaReference::Tensor* mask_tensor,
                                          TosaReference::Tensor* output_tensor,
                                          float scale)
    {
        // Get dimensions from tensors
        auto query_shape = query_tensor->getShape();
        auto key_shape = key_tensor->getShape();
        auto value_shape = value_tensor->getShape();
        
        int batch_size = query_shape[0];
        int num_heads = query_shape[1];
        int seq_len_q = query_shape[2];
        int seq_len_k = key_shape[2];
        int head_size = query_shape[3];

        // Cast tensors to Eigen tensors of rank 4 (batch, heads, seq_len, head_size)
        auto eigenQueryTensor = reinterpret_cast<TosaReference::TensorTemplate<Eigen::Tensor<float, 4>>*>(query_tensor);
        auto eigenKeyTensor = reinterpret_cast<TosaReference::TensorTemplate<Eigen::Tensor<float, 4>>*>(key_tensor);
        auto eigenValueTensor = reinterpret_cast<TosaReference::TensorTemplate<Eigen::Tensor<float, 4>>*>(value_tensor);
        auto eigenOutputTensor = reinterpret_cast<TosaReference::TensorTemplate<Eigen::Tensor<float, 4>>*>(output_tensor);

        // Get references to tensors
        auto& query = eigenQueryTensor->getTensor();
        auto& key = eigenKeyTensor->getTensor();
        auto& value = eigenValueTensor->getTensor();
        auto& output = eigenOutputTensor->getTensor();

        // Implementation of scaled dot-product attention
        // For each batch and head:
        // 1. Compute attention scores: Q * K^T
        // 2. Scale attention scores
        // 3. Apply mask (if provided)
        // 4. Apply softmax
        // 5. Compute weighted sum with values: softmax(QK^T) * V

        // Process each batch and head individually
        for (int b = 0; b < batch_size; b++) {
            for (int h = 0; h < num_heads; h++) {
                // 1. Compute attention scores: Q * K^T (seq_len_q x seq_len_k)
                Eigen::Tensor<float, 2> attention_scores(seq_len_q, seq_len_k);
                
                // Manual matrix multiplication Q * K^T
                for (int i = 0; i < seq_len_q; i++) {
                    for (int j = 0; j < seq_len_k; j++) {
                        float score = 0.0f;
                        for (int k = 0; k < head_size; k++) {
                            score += query(b, h, i, k) * key(b, h, j, k);
                        }
                        attention_scores(i, j) = score;
                    }
                }
                
                // 2. Scale attention scores
                attention_scores = attention_scores * scale;
                
                // 3. Apply mask if provided
                if (mask_tensor != nullptr) {
                    auto eigenMaskTensor = reinterpret_cast<TosaReference::TensorTemplate<Eigen::Tensor<float, 4>>*>(mask_tensor);
                    auto& mask = eigenMaskTensor->getTensor();
                    
                    // Apply mask - set masked positions to large negative value
                    for (int i = 0; i < seq_len_q; i++) {
                        for (int j = 0; j < seq_len_k; j++) {
                            if (mask(b, h, i, j) == 0.0f) { // 0 means masked position
                                attention_scores(i, j) = -10000.0f; // Large negative value
                            }
                        }
                    }
                }
                
                // 4. Apply softmax
                // First find the maximum value in each row for numerical stability
                Eigen::Tensor<float, 1> row_max(seq_len_q);
                for (int i = 0; i < seq_len_q; i++) {
                    float max_val = -std::numeric_limits<float>::infinity();
                    for (int j = 0; j < seq_len_k; j++) {
                        max_val = std::max(max_val, attention_scores(i, j));
                    }
                    row_max(i) = max_val;
                }
                
                // Compute softmax: exp(x - max) / sum(exp(x - max))
                Eigen::Tensor<float, 2> softmax_scores(seq_len_q, seq_len_k);
                Eigen::Tensor<float, 1> row_sum(seq_len_q);
                row_sum.setZero();
                
                // Compute exp(x - max) and row sums
                for (int i = 0; i < seq_len_q; i++) {
                    for (int j = 0; j < seq_len_k; j++) {
                        float exp_val = std::exp(attention_scores(i, j) - row_max(i));
                        softmax_scores(i, j) = exp_val;
                        row_sum(i) += exp_val;
                    }
                }
                
                // Normalize by row sums
                for (int i = 0; i < seq_len_q; i++) {
                    for (int j = 0; j < seq_len_k; j++) {
                        softmax_scores(i, j) /= row_sum(i);
                    }
                }
                
                // 5. Compute weighted sum with values: softmax(QK^T) * V
                // For each query position and head dimension
                for (int i = 0; i < seq_len_q; i++) {
                    for (int d = 0; d < value_shape[3]; d++) {
                        float weighted_sum = 0.0f;
                        for (int j = 0; j < seq_len_k; j++) {
                            weighted_sum += softmax_scores(i, j) * value(b, h, j, d);
                        }
                        output(b, h, i, d) = weighted_sum;
                    }
                }
            }
        }
        
        return 0;
    }

    std::string _domain_name;
    std::string _operator_name;
    std::string _version;
};

CustomOpInterface* createAttentionOp()
{
    std::string domain_name = "TosaMlirCustom";
    std::string operator_name = "Attention";
    std::string version = "1.0";
    CustomOpInterface* customOp_ptr = new AttentionOp(domain_name, operator_name, version);

    return customOp_ptr;
}

extern "C" TOSA_EXPORT int getCustomOpCreationFuncs(registration_callback_t registration_func)
{
    std::string domain_name = "TosaMlirCustom";
    std::string operator_name = "Attention";
    return registration_func(domain_name, operator_name, &createAttentionOp);
}

} // namespace TosaReference

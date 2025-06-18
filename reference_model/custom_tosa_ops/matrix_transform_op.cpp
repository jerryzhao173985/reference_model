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
 * @brief Custom Matrix Transform Operation
 * 
 * This operation provides specialized matrix transformations for ML workloads.
 * Supports operations like:
 * - Reshape (specialized for matrices)
 * - Transpose
 * - Matrix packing for efficient computation
 * - Special transformations for attention mechanisms
 */
class MatrixTransformOp : public CustomOpInterface
{
public:
    MatrixTransformOp() = default;
    MatrixTransformOp(std::string& domain_name, std::string& operator_name, std::string& version)
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

        // Get the transform type from attributes
        std::string transform_type = attrs.value("transform_type", "transpose");

        // Check if we have at least one input and output tensor
        if (input_tensors.empty() || output_tensors.empty()) {
            std::cerr << "Matrix transform requires at least one input and output tensor" << std::endl;
            return 1;
        }

        auto input_tensor = input_tensors[0];
        auto output_tensor = output_tensors[0];

        // Basic validation of tensor ranks
        auto input_shape = input_tensor->getShape();
        auto output_shape = output_tensor->getShape();

        // Only float types supported for now
        if (input_tensor->getDType() != DType::FLOAT32) {
            std::cerr << "Matrix transform currently only supports FLOAT32 tensors" << std::endl;
            return 1;
        }

        // For each transform type, delegate to the appropriate function
        if (transform_type == "transpose") {
            return evaluateTranspose(input_tensor, output_tensor, attrs);
        } else if (transform_type == "reshape") {
            return evaluateReshape(input_tensor, output_tensor, attrs);
        } else if (transform_type == "pack") {
            return evaluatePack(input_tensor, output_tensor, attrs);
        } else {
            std::cerr << "Unknown transform type: " << transform_type << std::endl;
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

    ~MatrixTransformOp(){}

private:
    int evaluateTranspose(TosaReference::Tensor* input_tensor, 
                          TosaReference::Tensor* output_tensor,
                          const json& attrs)
    {
        using TIn = Eigen::Tensor<float, 4>;
        using TOut = Eigen::Tensor<float, 4>;

        // Get permutation from attributes or use default
        std::vector<int32_t> perm;
        if (attrs.contains("perm")) {
            perm = attrs["perm"].get<std::vector<int32_t>>();
        } else {
            // Default permutation for matrix transpose
            auto rank = input_tensor->getRank();
            perm.resize(rank);
            for (int i = 0; i < rank; i++) {
                perm[i] = i;
            }
            // Swap last two dimensions for standard matrix transpose
            if (rank >= 2) {
                std::swap(perm[rank-1], perm[rank-2]);
            }
        }

        // Get the rank of the input tensor
        auto rank = input_tensor->getRank();

        // Validate permutation
        if (perm.size() != rank) {
            std::cerr << "Permutation size does not match tensor rank" << std::endl;
            return 1;
        }

        // Handle based on tensor rank
        switch (rank) {
            case 1: {
                auto eigenInputTensor = reinterpret_cast<TosaReference::TensorTemplate<Eigen::Tensor<float, 1>>*>(input_tensor);
                auto eigenOutputTensor = reinterpret_cast<TosaReference::TensorTemplate<Eigen::Tensor<float, 1>>*>(output_tensor);
                // For rank 1, transpose is a no-op
                eigenOutputTensor->getTensor() = eigenInputTensor->getTensor();
                break;
            }
            case 2: {
                auto eigenInputTensor = reinterpret_cast<TosaReference::TensorTemplate<Eigen::Tensor<float, 2>>*>(input_tensor);
                auto eigenOutputTensor = reinterpret_cast<TosaReference::TensorTemplate<Eigen::Tensor<float, 2>>*>(output_tensor);
                // Use Eigen's shuffle to implement the permutation
                Eigen::array<int, 2> shuffleArray = {perm[0], perm[1]};
                eigenOutputTensor->getTensor() = eigenInputTensor->getTensor().shuffle(shuffleArray);
                break;
            }
            case 3: {
                auto eigenInputTensor = reinterpret_cast<TosaReference::TensorTemplate<Eigen::Tensor<float, 3>>*>(input_tensor);
                auto eigenOutputTensor = reinterpret_cast<TosaReference::TensorTemplate<Eigen::Tensor<float, 3>>*>(output_tensor);
                // Use Eigen's shuffle to implement the permutation
                Eigen::array<int, 3> shuffleArray = {perm[0], perm[1], perm[2]};
                eigenOutputTensor->getTensor() = eigenInputTensor->getTensor().shuffle(shuffleArray);
                break;
            }
            case 4: {
                auto eigenInputTensor = reinterpret_cast<TosaReference::TensorTemplate<Eigen::Tensor<float, 4>>*>(input_tensor);
                auto eigenOutputTensor = reinterpret_cast<TosaReference::TensorTemplate<Eigen::Tensor<float, 4>>*>(output_tensor);
                // Use Eigen's shuffle to implement the permutation
                Eigen::array<int, 4> shuffleArray = {perm[0], perm[1], perm[2], perm[3]};
                eigenOutputTensor->getTensor() = eigenInputTensor->getTensor().shuffle(shuffleArray);
                break;
            }
            default:
                std::cerr << "Transpose only supports tensors with rank <= 4" << std::endl;
                return 1;
        }

        return 0;
    }

    int evaluateReshape(TosaReference::Tensor* input_tensor, 
                         TosaReference::Tensor* output_tensor,
                         const json& attrs)
    {
        // Check if shapes are compatible (same total number of elements)
        auto input_shape = input_tensor->getShape();
        auto output_shape = output_tensor->getShape();

        int64_t input_size = 1;
        for (auto dim : input_shape) {
            input_size *= dim;
        }

        int64_t output_size = 1;
        for (auto dim : output_shape) {
            output_size *= dim;
        }

        if (input_size != output_size) {
            std::cerr << "Reshape requires input and output tensors to have the same number of elements" << std::endl;
            return 1;
        }

        // Reshape is essentially a memory copy operation with same elements but different layout
        // Get input data as flat array of float values
        auto eigenInputTensor = reinterpret_cast<TosaReference::TensorTemplate<Eigen::Tensor<float, 1>>*>(input_tensor);
        auto eigenOutputTensor = reinterpret_cast<TosaReference::TensorTemplate<Eigen::Tensor<float, 1>>*>(output_tensor);

        // Create a flat view of the input tensor
        Eigen::Tensor<float, 1> inputFlat = Eigen::TensorMap<Eigen::Tensor<float, 1>>(
            eigenInputTensor->getTensor().data(), input_size);

        // Create a flat view of the output tensor
        Eigen::Tensor<float, 1> outputFlat = Eigen::TensorMap<Eigen::Tensor<float, 1>>(
            eigenOutputTensor->getTensor().data(), output_size);

        // Copy data from input to output
        outputFlat = inputFlat;

        return 0;
    }

    int evaluatePack(TosaReference::Tensor* input_tensor, 
                      TosaReference::Tensor* output_tensor,
                      const json& attrs)
    {
        // Matrix packing operation for efficient computation
        // Typically used for packing matrices for GEMM operations

        // Get packing type from attributes
        std::string pack_type = attrs.value("pack_type", "column_major");
        int block_size = attrs.value("block_size", 8); // Default block size

        auto input_shape = input_tensor->getShape();
        auto output_shape = output_tensor->getShape();

        // Ensure input is a matrix (rank 2)
        if (input_shape.size() != 2) {
            std::cerr << "Matrix packing requires a rank-2 input tensor" << std::endl;
            return 1;
        }

        // Get dimensions
        int rows = input_shape[0];
        int cols = input_shape[1];

        // Convert tensors to appropriate Eigen types
        auto eigenInputTensor = reinterpret_cast<TosaReference::TensorTemplate<Eigen::Tensor<float, 2>>*>(input_tensor);
        auto eigenOutputTensor = reinterpret_cast<TosaReference::TensorTemplate<Eigen::Tensor<float, 2>>*>(output_tensor);
        
        // Get references to Eigen tensors
        auto& inputTensor = eigenInputTensor->getTensor();
        auto& outputTensor = eigenOutputTensor->getTensor();

        // Perform packing based on type
        if (pack_type == "column_major") {
            // Simple column-major packing (essentially a transpose)
            Eigen::array<int, 2> shuffleArray = {1, 0}; // Transpose dimensions
            outputTensor = inputTensor.shuffle(shuffleArray);
        } 
        else if (pack_type == "block") {
            // Block-based packing for efficient matrix multiplication
            // This is a simplified implementation - real implementations would be more complex
            
            // Calculate number of block rows and columns
            int block_rows = (rows + block_size - 1) / block_size;
            int block_cols = (cols + block_size - 1) / block_size;
            
            // Ensure output shape is correct for blocked format
            if (output_shape[0] != block_rows * block_cols && 
                output_shape[1] != block_size * block_size) {
                std::cerr << "Output shape not compatible with block packing" << std::endl;
                return 1;
            }
            
            // Initialize output to zeros
            outputTensor.setZero();
            
            // Pack input into blocks
            for (int i = 0; i < block_rows; i++) {
                for (int j = 0; j < block_cols; j++) {
                    int block_idx = i * block_cols + j;
                    
                    // Copy data from input block to output block
                    for (int bi = 0; bi < block_size; bi++) {
                        for (int bj = 0; bj < block_size; bj++) {
                            int row = i * block_size + bi;
                            int col = j * block_size + bj;
                            
                            // Check bounds
                            if (row < rows && col < cols) {
                                // Linear index in output
                                int out_idx = block_idx * block_size * block_size + bi * block_size + bj;
                                
                                // Set output element
                                outputTensor(out_idx / (block_size * block_size), 
                                             out_idx % (block_size * block_size)) = 
                                    inputTensor(row, col);
                            }
                        }
                    }
                }
            }
        }
        else {
            std::cerr << "Unknown packing type: " << pack_type << std::endl;
            return 1;
        }

        return 0;
    }

    std::string _domain_name;
    std::string _operator_name;
    std::string _version;
};

CustomOpInterface* createMatrixTransformOp()
{
    std::string domain_name = "TosaMlirCustom";
    std::string operator_name = "MatrixTransform";
    std::string version = "1.0";
    CustomOpInterface* customOp_ptr = new MatrixTransformOp(domain_name, operator_name, version);

    return customOp_ptr;
}

extern "C" TOSA_EXPORT int getCustomOpCreationFuncs(registration_callback_t registration_func)
{
    std::string domain_name = "TosaMlirCustom";
    std::string operator_name = "MatrixTransform";
    return registration_func(domain_name, operator_name, &createMatrixTransformOp);
}

} // namespace TosaReference

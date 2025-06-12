// Copyright (c) 2024, Test Example
// This test program demonstrates the reference model's graph execution mechanism
// by building a small computational graph: input -> ABS -> ADD -> output
//
// Build this as part of the reference model test suite or compile with:
//   g++ -std=c++17 -I./reference_model/src -I./include -I./third_party/eigen -I./third_party/flatbuffers/include \
//       -DEIGEN_NO_DEBUG -DEIGEN_STACK_ALLOCATION_LIMIT=0 \
//       test_graph_example.cpp <link with reference model libraries>

#include "subgraph_traverser.h"
#include "graph_node.h"
#include "ops/op_factory.h"
#include "ops/ewise_unary.h"
#include "ops/ewise_binary.h"
#include "tensor.h"
#include "tosa_serialization_handler.h"
#include "func_config.h"
#include "func_debug.h"
#include <iostream>
#include <memory>
#include <vector>

// Global configuration structures
func_debug_t g_func_debug = {};  // Use default initialization
func_config_t g_func_config = {};

using namespace TosaReference;
using namespace tosa;

// Helper function to create an operator
std::unique_ptr<TosaSerializationOperator> createOperator(
    tosa::Op opType,
    tosa::TosaAttributeBase* attr,
    const std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs)
{
    // Get the attribute type for this operator
    tosa::Attribute attrType = tosa::Attribute::Attribute_NONE;
    
    // For operators that have attributes, set the appropriate type
    if (attr != nullptr) {
        // Determine attribute type based on the actual attribute object
        // For this example, we'll use NONE since ABS and ADD don't have attributes
        attrType = tosa::Attribute::Attribute_NONE;
    }
    
    // Create operator with location
    TosaOpLocation loc;
    loc.text = "test_graph_example.cpp";  // Set location text
    return std::make_unique<TosaSerializationOperator>(opType, attrType, attr, inputs, outputs, loc);
}

int main() {
    std::cout << "\n=== TOSA Reference Model Graph Execution Example ===\n\n";

    // Initialize debug configuration with proper output
    g_func_debug.func_debug_verbosity = DEBUG_VERB_HIGH;  // Enable debug output
    g_func_debug.func_debug_mask = DEBUG_ALL;             // Enable all debug units
    g_func_debug.func_debug_file = stdout;                // Send debug to stdout
    
    // Initialize function configuration with proper level
    g_func_config.tosa_level = func_config_t::EIGHTK;  // Use the EIGHTK level

    try {
        // Create TosaSerializationHandler first
        std::cout << "Creating TosaSerializationHandler...\n";
        TosaSerializationHandler tsh;
        
        // Create a region
        std::cout << "Creating region...\n";
        auto region = std::make_unique<TosaSerializationRegion>("main_region");
        
        // Now build the block directly in the region
        {
            // Define tensor shapes
            std::vector<int> input_shape = {4};  // 1D tensor with 4 elements

            // Create tensors for the graph
            std::cout << "Creating tensors...\n";
            std::vector<std::unique_ptr<TosaSerializationTensor>> tensors;
            tensors.push_back(std::make_unique<TosaSerializationTensor>(
                "input1", input_shape, tosa::DType::DType_FP32, std::vector<uint8_t>()));
            tensors.push_back(std::make_unique<TosaSerializationTensor>(
                "input2", input_shape, tosa::DType::DType_FP32, std::vector<uint8_t>()));
            tensors.push_back(std::make_unique<TosaSerializationTensor>(
                "temp", input_shape, tosa::DType::DType_FP32, std::vector<uint8_t>()));
            tensors.push_back(std::make_unique<TosaSerializationTensor>(
                "output", input_shape, tosa::DType::DType_FP32, std::vector<uint8_t>()));

            // Create operators
            std::cout << "Creating operators...\n";
            std::vector<std::unique_ptr<TosaSerializationOperator>> operators;
            // 1. ABS operator: input1 -> temp
            operators.push_back(createOperator(tosa::Op::Op_ABS, nullptr, {"input1"}, {"temp"}));
            // 2. ADD operator: temp + input2 -> output
            operators.push_back(createOperator(tosa::Op::Op_ADD, nullptr, {"temp", "input2"}, {"output"}));

            // Input and output tensor names
            std::vector<std::string> inputs = {"input1", "input2"};
            std::vector<std::string> outputs = {"output"};

            // Create basic block with empty shapes vector
            std::string block_name = "main";
            std::string region_name = "";  // Empty region name for main block
            std::vector<std::unique_ptr<TosaSerializationShape>> shapes;  // Empty shapes
            
            // Add block to region
            std::cout << "Creating basic block...\n";
            region->GetBlocks().push_back(std::make_unique<TosaSerializationBasicBlock>(
                block_name, region_name, 
                std::move(operators), std::move(tensors), 
                std::move(shapes), inputs, outputs));
        }
        
        // Add region to handler
        std::cout << "Adding region to handler...\n";
        tsh.GetRegions().push_back(std::move(region));

        // Get pointers before creating traverser
        auto main_region = tsh.GetMainRegion();
        if (!main_region) {
            std::cerr << "Failed to get main region\n";
            return 1;
        }
        std::cout << "Got main region: " << main_region->GetName() << "\n";
        
        auto& blocks = main_region->GetBlocks();
        if (blocks.empty()) {
            std::cerr << "No blocks in main region\n";
            return 1;
        }
        std::cout << "Got " << blocks.size() << " blocks\n";
        
        auto main_block = blocks[0].get();
        if (!main_block) {
            std::cerr << "Main block is null\n";
            return 1;
        }
        std::cout << "Got main block: " << main_block->GetName() << "\n";

        // Create SubgraphTraverser
        std::cout << "\nCreating SubgraphTraverser...\n";
        SubgraphTraverser traverser(main_block, &tsh, nullptr);

        // Initialize the graph
        std::cout << "Initializing graph...\n";
        if (traverser.initializeGraph()) {
            std::cerr << "Failed to initialize graph\n";
            return 1;
        }
        std::cout << "Graph initialized successfully\n";

        // Link tensors and nodes
        std::cout << "Linking tensors and nodes...\n";
        if (traverser.linkTensorsAndNodes()) {
            std::cerr << "Failed to link tensors and nodes\n";
            return 1;
        }
        std::cout << "Tensors and nodes linked successfully\n";

        // Validate graph
        std::cout << "Validating graph...\n";
        if (traverser.validateGraph()) {
            std::cerr << "Failed to validate graph\n";
            return 1;
        }
        std::cout << "Graph validated successfully\n";

        // Allocate all tensors
        std::cout << "\nAllocating tensors...\n";
        std::vector<std::string> all_tensors = {"input1", "input2", "temp", "output"};
        for (const auto& tensor_name : all_tensors) {
            std::cout << "  Allocating tensor: " << tensor_name << "\n";
            if (traverser.allocateTensor(tensor_name)) {
                std::cerr << "  Failed to allocate tensor: " << tensor_name << "\n";
                return 1;
            }
        }
        std::cout << "All tensors allocated successfully\n";

        // Set input values
        std::cout << "\nSetting input values...\n";
        
        // Get the actual Tensor objects from the traverser
        Tensor* input1 = traverser.getInputTensorByName("input1");
        Tensor* input2 = traverser.getInputTensorByName("input2");
        
        if (!input1) {
            std::cerr << "Failed to get input1 tensor\n";
            return 1;
        }
        if (!input2) {
            std::cerr << "Failed to get input2 tensor\n";
            return 1;
        }
        std::cout << "Got input tensors successfully\n";

        // Set input1 values: [-2.0, -1.0, 1.0, 2.0]
        std::cout << "Setting input1 values...\n";
        float input1_data[] = {-2.0f, -1.0f, 1.0f, 2.0f};
        if (input1->setTensorValueFloat(4, input1_data) != 0) {
            std::cerr << "Failed to set input1 values\n";
            return 1;
        }

        // Set input2 values: [1.0, 1.0, 1.0, 1.0]
        std::cout << "Setting input2 values...\n";
        float input2_data[] = {1.0f, 1.0f, 1.0f, 1.0f};
        if (input2->setTensorValueFloat(4, input2_data) != 0) {
            std::cerr << "Failed to set input2 values\n";
            return 1;
        }

        // Mark inputs as valid
        std::cout << "Marking inputs as valid...\n";
        input1->setIsValid();
        input2->setIsValid();

        // Run the graph
        std::cout << "\nRunning graph execution...\n";
        if (traverser.evaluateAll()) {
            std::cerr << "Failed to evaluate graph\n";
            return 1;
        }
        
        // // Instead of evaluateAll, we can run the graph step by step
        // // Step 6: Execute the graph step by step
        // std::cout << "\n--- Executing graph ---" << std::endl;
        // int step = 0;
        // while (!traverser.isFullyEvaluated())
        // {
        //     std::cout << "\nStep " << step++ << ": ";
            
        //     // Show what's ready to execute
        //     traverser.dumpNextNodeList(stdout);
            
        //     // Execute next node
        //     if (traverser.evaluateNextNode())
        //     {
        //         std::cerr << "Failed to evaluate node!" << std::endl;
        //         return 1;
        //     }
        
        std::cout << "Graph execution completed successfully\n";

        // Get output values
        std::cout << "\nGetting output values...\n";
        Tensor* output_tensor = traverser.getOutputTensorByName("output");
        if (!output_tensor) {
            std::cerr << "Failed to get output tensor\n";
            return 1;
        }

        // Read output data
        float output_data[4];
        if (output_tensor->getTensorValueFloat(4, output_data) != 0) {
            std::cerr << "Failed to get output data\n";
            return 1;
        }

        // Step 8: Show graph structure
        std::cout << "\n=== Graph Structure ===" << std::endl;
        traverser.dumpGraph(stdout);
        
        // Display results
        std::cout << "\n=== Results ===\n";
        std::cout << "Input1: [-2.0, -1.0, 1.0, 2.0]\n";
        std::cout << "Input2: [ 1.0,  1.0, 1.0, 1.0]\n";
        std::cout << "Expected: [ 3.0,  2.0, 2.0, 3.0] (abs(input1) + input2)\n";
        std::cout << "Actual:   [";
        for (int i = 0; i < 4; i++) {
            std::cout << " " << output_data[i];
            if (i < 3) std::cout << ",";
        }
        std::cout << " ]\n";

        // Verify results
        float expected[] = {3.0f, 2.0f, 2.0f, 3.0f};
        bool success = true;
        for (int i = 0; i < 4; i++) {
            if (std::abs(output_data[i] - expected[i]) > 1e-6) {
                success = false;
                break;
            }
        }

        std::cout << "\nTest " << (success ? "PASSED" : "FAILED") << "!\n";

        return success ? 0 : 1;

    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "Unknown exception caught\n";
        return 1;
    }
}
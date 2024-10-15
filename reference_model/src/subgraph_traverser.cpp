
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

#include "subgraph_traverser.h"
#include "arith_util.h"
#include "tosa_model_types.h"
#include <cinttypes>

#ifndef SUBGRAPH_ERROR_IF
#define SUBGRAPH_ERROR_IF(COND, fmt, ...)                                                                              \
    if ((COND))                                                                                                        \
    {                                                                                                                  \
        if (this->getGraphStatus() != GraphStatus::TOSA_UNPREDICTABLE)                                                 \
        {                                                                                                              \
            this->setGraphStatus(GraphStatus::TOSA_ERROR);                                                             \
        }                                                                                                              \
        fprintf(g_func_debug.func_debug_file, COL_FATAL("SUBGRAPH_ERROR_IF() fails AT %s:%d %s(): (%s)\n"), __FILE__,  \
                __LINE__, __func__, #COND);                                                                            \
        fprintf(g_func_debug.func_debug_file, COL_FATAL(fmt) "\n", ##__VA_ARGS__);                                     \
        func_print_backtrace(g_func_debug.func_debug_file);                                                            \
        return 1;                                                                                                      \
    }
#endif

using namespace TosaReference;
using namespace Eigen;
using namespace tosa;

SubgraphTraverser::SubgraphTraverser(TosaSerializationBasicBlock* _block,
                                     TosaSerializationHandler* _tsh,
                                     SubgraphTraverser* _parent_sgt)
{

    graph_status = GraphStatus::TOSA_VALID;
    block        = _block;

    tsh        = _tsh;
    parent_sgt = _parent_sgt;
    tensors.clear();
    nodes.clear();
    nextNodeList.clear();
}

SubgraphTraverser::~SubgraphTraverser()
{
    nextNodeList.clear();

    for (GraphNode* n : nodes)
    {
        delete n;
    }
    nodes.clear();

    for (TosaReference::Tensor* t : tensors)
    {
        if (t->getIsVariable() && parent_sgt)
        {
            // variable tensors are owned by top level sgt
            continue;
        }
        if (t->is_allocated())
        {
            t->deallocate();
        }
        delete t;
    }
    tensors.clear();
}

int SubgraphTraverser::getNumInputTensors() const
{
    return static_cast<int>(inputTensors.size());
}

TosaReference::Tensor* SubgraphTraverser::getInputTensor(const unsigned int idx) const
{
    return inputTensors[idx];
}

TosaReference::Tensor* SubgraphTraverser::getInputTensorByName(const std::string name) const
{
    for (auto t : inputTensors)
    {
        if (t->getName() == name)
        {
            return t;
        }
    }

    return nullptr;
}

int SubgraphTraverser::getNumOutputTensors() const
{
    return static_cast<int>(outputTensors.size());
}

TosaReference::Tensor* SubgraphTraverser::getOutputTensor(const unsigned int idx) const
{
    return outputTensors[idx];
}

TosaReference::Tensor* SubgraphTraverser::getOutputTensorByName(const std::string name) const
{
    for (auto t : outputTensors)
    {
        if (t->getName() == name)
        {
            return t;
        }
    }

    return nullptr;
}

int SubgraphTraverser::getNumVariableTensors() const
{
    return static_cast<int>(variableTensors.size());
}

TosaReference::Tensor* SubgraphTraverser::getVariableTensor(const unsigned int idx) const
{
    return variableTensors[idx];
}

// find variable tensor by name in top level sgt's @a variableTensors
TosaReference::Tensor* SubgraphTraverser::getVariableTensorByName(const std::string name) const
{
    // variable tensors are owned by top level sgt
    if (parent_sgt)
    {
        return parent_sgt->getVariableTensorByName(name);
    }

    for (auto t : variableTensors)
    {
        if (t->getName() == name)
        {
            return t;
        }
    }

    return nullptr;
}

// add variable tensor to top level sgt's @a variableTensors
int SubgraphTraverser::registerVariableTensor(Tensor* tensor)
{
    SUBGRAPH_ERROR_IF(!tensor->getIsVariable(),
                      "SubgraphTraverser::registerVariableTensor(): tensor %s is not a variable",
                      tensor->getName().c_str());
    // variable tensors are owned by top level sgt
    if (parent_sgt)
    {
        return parent_sgt->registerVariableTensor(tensor);
    }
    variableTensors.push_back(tensor);
    return 0;
}

TosaSerializationTensor* SubgraphTraverser::getSerializationTensorByName(const std::string& name) const
{
    for (auto ser_tensor : ser_tensor_vec)
    {
        if (ser_tensor->GetName() == name)
        {
            return ser_tensor;
        }
    }
    return nullptr;
}

TosaSerializationShape* SubgraphTraverser::getSerializationShapeByName(const std::string& name) const
{
    for (auto ser_shape : ser_shape_vec)
    {
        if (ser_shape->GetName() == name)
        {
            return ser_shape;
        }
    }
    return nullptr;
}

bool SubgraphTraverser::findDtypeAndRankByName(const std::string& name, TOSA_REF_TYPE& dtype, int32_t& rank) const
{
    if (auto ser_tensor = getSerializationTensorByName(name))
    {
        dtype = ConvertDType(ser_tensor->GetDtype());
        rank  = static_cast<int32_t>(ser_tensor->GetShape().size());
        return true;
    }
    if (auto ser_shape = getSerializationShapeByName(name))
    {
        // shape values: dtype is TOSA_REF_TYPE_SHAPE and rank is 1
        dtype = TOSA_REF_TYPE_SHAPE;
        rank  = 1;
        return true;
    }

    // name is not found
    dtype = TOSA_REF_TYPE_UNKNOWN;
    rank  = 0;

    return false;
}

int SubgraphTraverser::initializeGraph()
{
    uint64_t idx = 0;

    // Get all the serialized tensors from TosaSerializationHandler.
    if (tsh)
    {
        for (const auto& region : tsh->GetRegions())
        {
            for (const auto& block : region->GetBlocks())
            {
                for (const auto& ser_tensor : block->GetTensors())
                {
                    ser_tensor_vec.push_back(ser_tensor.get());
                }
                for (const auto& ser_shape : block->GetShapes())
                {
                    ser_shape_vec.push_back(ser_shape.get());
                }
            }
        }
    }
    else
    {
        for (const auto& ser_tensor : block->GetTensors())
        {
            ser_tensor_vec.push_back(ser_tensor.get());
        }
        for (const auto& ser_shape : block->GetShapes())
        {
            ser_shape_vec.push_back(ser_shape.get());
        }
    }

    std::vector<GraphNode*> non_const_node_vec;
    for (const auto& op : block->GetOperators())
    {
        // translated TosaSerializationOperator to GraphNode
        TOSA_REF_TYPE input_dtype  = TOSA_REF_TYPE_UNKNOWN;
        TOSA_REF_TYPE output_dtype = TOSA_REF_TYPE_UNKNOWN;
        TOSA_REF_TYPE weight_dtype = TOSA_REF_TYPE_UNKNOWN;
        TOSA_REF_TYPE bias_dtype   = TOSA_REF_TYPE_UNKNOWN;
        int32_t input_rank         = 0;
        int32_t output_rank        = 0;
        int32_t weight_rank        = 0;
        int32_t input_index        = -1;
        int32_t weight_index       = -1;
        int32_t bias_index         = -1;

        switch (op->GetOp())
        {
            case Op_CONV2D:
            case Op_CONV3D:
            case Op_DEPTHWISE_CONV2D:
            case Op_TRANSPOSE_CONV2D:
                input_index  = 0;
                weight_index = 1;
                bias_index   = 2;
                break;
            case Op_MATMUL:
                input_index  = 0;
                weight_index = 1;
                break;
            case Op_SELECT:
                input_index = 1;
                break;
            default:
                if (!op->GetInputTensorNames().empty())
                    input_index = 0;
                break;
        }

        if (input_index != -1)
        {
            SUBGRAPH_ERROR_IF(
                (size_t)input_index >= op->GetInputTensorNames().size(),
                "SubgraphTraverser::initializeGraph(): Op=%s, input_index %d must be within [0, num_input - 1]",
                EnumNamesOp()[op->GetOp()], input_index);

            std::string input_name = op->GetInputTensorNames()[static_cast<size_t>(input_index)];
            bool found             = findDtypeAndRankByName(input_name, input_dtype, input_rank);
            SUBGRAPH_ERROR_IF(
                !found,
                "SubgraphTraverser::initializeGraph(): fail to get input tensor %s from TosaSerializationHandler",
                input_name.c_str());
        }

        if (weight_index != -1)
        {
            SUBGRAPH_ERROR_IF(
                (size_t)weight_index >= op->GetInputTensorNames().size(),
                "SubgraphTraverser::initializeGraph(): Op=%s, weight_index %d must be within [0, num_input - 1]",
                EnumNamesOp()[op->GetOp()], weight_index);
            std::string weight_name = op->GetInputTensorNames()[static_cast<size_t>(weight_index)];
            bool found              = findDtypeAndRankByName(weight_name, weight_dtype, weight_rank);
            SUBGRAPH_ERROR_IF(
                !found,
                "SubgraphTraverser::initializeGraph(): fail to get weight tensor %s from TosaSerializationHandler",
                weight_name.c_str());
        }

        SUBGRAPH_ERROR_IF(op->GetOutputTensorNames().size() == 0,
                          "SubgraphTraverser::initializeGraph(): Op=%s must have at least one output tensor.",
                          EnumNamesOp()[op->GetOp()]);
        std::string output_name = op->GetOutputTensorNames()[0];
        {
            bool found = findDtypeAndRankByName(output_name, output_dtype, output_rank);
            SUBGRAPH_ERROR_IF(
                !found,
                "SubgraphTraverser::initializeGraph(): fail to get output tensor %s from TosaSerializationHandler",
                output_name.c_str());
        }
        if (bias_index != -1)
        {
            SUBGRAPH_ERROR_IF(
                (size_t)bias_index >= op->GetInputTensorNames().size(),
                "SubgraphTraverser::initializeGraph(): Op=%s, bias_index %d must be within [0, num_input - 1]",
                EnumNamesOp()[op->GetOp()], bias_index);
            std::string bias_name = op->GetInputTensorNames()[static_cast<size_t>(bias_index)];
            int32_t bias_rank;
            bool found = findDtypeAndRankByName(bias_name, bias_dtype, bias_rank);
            SUBGRAPH_ERROR_IF(
                !found,
                "SubgraphTraverser::initializeGraph(): fail to get bias tensor %s from TosaSerializationHandler",
                bias_name.c_str());

            SUBGRAPH_ERROR_IF(
                bias_dtype != output_dtype,
                "SubgraphTraverser::initializeGraph(): Op=%s, bias_dtype (%s) is different from output_dtype (%s)",
                EnumNamesOp()[op->GetOp()], EnumNameTOSAREFTYPE(bias_dtype), EnumNameTOSAREFTYPE(output_dtype));
        }

        DEBUG_INFO(GT, "Creating operator id_%03" PRIu64 ", %8s, %zu input tensors, %zu output tensors", idx,
                   EnumNamesOp()[op->GetOp()], op->GetInputTensorNames().size(), op->GetOutputTensorNames().size());

        GraphNode* node = nullptr;
        if (this->parent_sgt)
        {
            node = OpFactory::newOp(this->parent_sgt, tsh, op->GetOp(), op->GetAttribute(), idx, input_dtype,
                                    input_rank, output_dtype, output_rank, weight_dtype, weight_rank);
            node->setInMainBlock(false);
        }
        else
        {
            node = OpFactory::newOp(this, tsh, op->GetOp(), op->GetAttribute(), idx, input_dtype, input_rank,
                                    output_dtype, output_rank, weight_dtype, weight_rank);
            if (node)
            {
                node->setInMainBlock(true);
            }
        }

        if (!node)
        {
            if (weight_index == -1)
            {
                fprintf(g_func_debug.func_debug_file,
                        "SubgraphTraverser::initializeGraph(): OpFactory could not allocate op %8s input=(%s rank %d) "
                        "-> (%s rank %d)",
                        EnumNamesOp()[op->GetOp()], EnumNameTOSAREFTYPE(input_dtype), input_rank,
                        EnumNameTOSAREFTYPE(output_dtype), output_rank);
            }
            else
            {
                fprintf(g_func_debug.func_debug_file,
                        "SubgraphTraverser::initializeGraph(): OpFactory could not allocate op %8s input=(%s rank %d), "
                        "weight=(%s rank %d) -> (%s rank %d)",
                        EnumNamesOp()[op->GetOp()], EnumNameTOSAREFTYPE(input_dtype), input_rank,
                        EnumNameTOSAREFTYPE(weight_dtype), weight_rank, EnumNameTOSAREFTYPE(output_dtype), output_rank);
            }

            for (auto& ts : op->GetInputTensorNames())
            {
                fprintf(g_func_debug.func_debug_file, "SubgraphTraverser::initializeGraph(): Input: %s\n", ts.c_str());
            }

            for (auto& ts : op->GetOutputTensorNames())
            {
                fprintf(g_func_debug.func_debug_file, "SubgraphTraverser::initializeGraph(): Output: %s\n", ts.c_str());
            }
            SUBGRAPH_ERROR_IF(true, "SubgraphTraverser::initializeGraph(): Unsupported operation type or rank.");
        }

        // Elementwise operator might set TOSA_ERROR when registering lambda function when creating the op.
        // Check graph status after the op being constructed.
        SUBGRAPH_ERROR_IF(getGraphStatus() == GraphStatus::TOSA_ERROR,
                          "SubgraphTraverser::initializeGraph(): Op %8s triggered ERROR_IF() when constructing the op.",
                          EnumNamesOp()[op->GetOp()]);

        for (auto& name : op->GetInputTensorNames())
        {
            node->addInputName(name);
            used_tensor_name_set.insert(name);
        }

        for (auto name : op->GetOutputTensorNames())
        {
            node->addOutputName(name);
            used_tensor_name_set.insert(name);
        }

        addNode(node);

        // if node doesn't have any inputs (i.e. CONST)
        // it should be ready for evaluation
        if (op->GetInputTensorNames().empty() && !node->getOnNextNodeList())
        {
            addToNextNodeList(node);
        }
        else if (!node->getInMainBlock())
        {
            non_const_node_vec.push_back(node);
        }

        // Bug fix: add the ready node in main block for evaluation
        if (node->hasAllInputsReady() && !node->getOnNextNodeList() && !node->getEvaluated())
        {
            addToNextNodeList(node);
        }

        idx++;
    }

    for (const auto& ts : block->GetTensors())
    {
        addTensor(ts.get());
    }

    for (const auto& ts : block->GetShapes())
    {
        addShape(ts.get());
    }

    DEBUG_INFO(GT, "Enumerating block %s graph inputs", block->GetName().c_str());
    for (auto& input_name : block->GetInputs())
    {
        TosaReference::Tensor* tensor = findTensorByName(input_name);
        DEBUG_INFO(GT, "input tensor name=%s", input_name.c_str());
        if (tensor)
        {
            tensor->setIsSubgraphInput();
            inputTensors.push_back(tensor);
        }
        else
        {
            SUBGRAPH_ERROR_IF(true, "SubgraphTraverser::initializeGraph(): Failed to find input tensor by name %s",
                              input_name.c_str());
        }
    }

    DEBUG_INFO(GT, "Enumerating block %s graph outputs", block->GetName().c_str());
    for (auto& output_name : block->GetOutputs())
    {
        TosaReference::Tensor* tensor = findTensorByName(output_name);
        DEBUG_INFO(GT, "output tensor name=%s", output_name.c_str());
        if (tensor)
        {
            tensor->setIsSubgraphOutput();
            outputTensors.push_back(tensor);
        }
        else
        {
            SUBGRAPH_ERROR_IF(true, "SubgraphTraverser::initializeGraph(): Failed to find output tensor by name %s",
                              output_name.c_str());
        }
    }

    if (DEBUG_ENABLED(DEBUG_VERB_HIGH, GT))
    {
        dumpNextNodeList(g_func_debug.func_debug_file);
    }

    // If the node is not in mainblock and not const
    for (auto node : non_const_node_vec)
    {
        bool all_inputs_from_parent = true;
        for (std::string& name : node->getInputNames())
        {
            TosaReference::Tensor* t = findTensorByName(name);
            if (!t->getIsParentGraphOutput())
            {
                all_inputs_from_parent = false;
            }
        }
        // In the children block, when a node has all its inputs from parent
        // block, we have to manually add this node to the evaluation list
        if (all_inputs_from_parent && !node->getOnNextNodeList())
        {
            addToNextNodeList(node);
        }
    }
    return 0;
}

int SubgraphTraverser::allocateInputTensors()
{
    auto input_tensor_names_vec = block->GetInputs();

    for (auto input_tensor_name : input_tensor_names_vec)
    {
        this->allocateTensor(input_tensor_name);
    }

    // allocate variable tensors if not already allocated
    for (const auto& ts : block->GetTensors())
    {
        if (ts->GetVariable())
        {
            TosaReference::Tensor* tensor = findTensorByName(ts->GetName());
            SUBGRAPH_ERROR_IF(!tensor, "SubgraphTraverser::allocateInputTensors(): can't find tensor %s.",
                              ts->GetName().c_str());
            if (!tensor->is_allocated())
            {
                DEBUG_INFO(GT, "Is a VariableTensor %s", ts->GetName().c_str());
                this->allocateTensor(ts->GetName());
            }
        }
    }

    return 0;
}

int SubgraphTraverser::allocateTensor(std::string name)
{
    std::vector<int32_t> shape;
    std::vector<uint8_t> data;
    DType dtype      = DType_UNKNOWN;
    bool is_variable = false;

    if (auto ser_tensor = getSerializationTensorByName(name))
    {
        shape       = ser_tensor->GetShape();
        data        = ser_tensor->GetData();
        dtype       = ser_tensor->GetDtype();
        is_variable = ser_tensor->GetVariable();
    }
    else if (auto ser_shape = getSerializationShapeByName(name))
    {
        if (ser_shape->GetRank() == 0)
        {
            // special case: rank 0 has empty shape
            shape = {};
        }
        else
        {
            shape = { static_cast<int32_t>(ser_shape->GetRank()) };
        }
        data        = ser_shape->GetData();
        dtype       = DType_SHAPE;
        is_variable = false;
    }
    else
    {
        SUBGRAPH_ERROR_IF(false, "SubgraphTraverser::allocateTensor(): can't find serialization tensor or shape %s.",
                          name.c_str());
    }

    // Bail out if tensor is used and any of its dimension is invalid.
    auto got = used_tensor_name_set.find(name);
    if (got != used_tensor_name_set.end())
    {
        uint32_t elements = 1;
        for (auto& dim : shape)
        {
            if (dim <= 0)
            {
                DEBUG_INFO(GT, "Failed to allocate tensor %s with invalid dimension of %d", name.c_str(), dim);
                this->setGraphStatus(GraphStatus::TOSA_UNPREDICTABLE);
                return 1;
            }
            if (dim > static_cast<int32_t>(TOSA_MAX_TENSOR_SIZE / elements))
            {
                // Size greather than maximum defined in spec
                DEBUG_INFO(GT, "Tensor %s size is greater than allowed maximum", name.c_str());
                this->setGraphStatus(GraphStatus::TOSA_UNPREDICTABLE);
                return 1;
            }
        }
    }

    TosaReference::Tensor* tensor = findTensorByName(name);
    SUBGRAPH_ERROR_IF(!tensor, "SubgraphTraverser::allocateTensor(): can't find tensor %s.", name.c_str());

    DEBUG_INFO(GT, "Allocating tensor %s", tensor->getName().c_str());
    if (tensor->allocate())
    {
        FATAL_ERROR("Failed to allocate tensor %s", tensor->getName().c_str());
    }

    // set valid for constant tensors:
    if ((shape.empty() && dtype == DType_SHAPE))
    {
        // corner case: const_shape {} has no data
        tensor->setIsValid();
    }
    if (!data.empty())
    {
        if (is_variable && g_func_config.initialize_variable_tensor_from_numpy)
            return 0;
        DEBUG_INFO(GT, "Setting data for tensor %s", tensor->getName().c_str());
        auto serialization_dtype = dtype;
        switch (serialization_dtype)
        {
            case DType_INT4: {
                std::vector<int8_t> i4_data;
                TosaSerializationHandler::ConvertU8toI4(data, tensor->getElementCount(), i4_data);
                std::vector<int32_t> i32_data(i4_data.begin(), i4_data.end());
                tensor->setTensorValueInt32(i32_data.size(), i32_data.data());
            }
            break;
            case DType_INT8: {
                std::vector<int8_t> i8_data;
                TosaSerializationHandler::ConvertU8toI8(data, tensor->getElementCount(), i8_data);
                std::vector<int32_t> i32_data(i8_data.begin(), i8_data.end());
                tensor->setTensorValueInt32(i32_data.size(), i32_data.data());
            }
            break;
            case DType_INT16: {
                std::vector<int16_t> i16_data;
                TosaSerializationHandler::ConvertU8toI16(data, tensor->getElementCount(), i16_data);
                std::vector<int32_t> i32_data(i16_data.begin(), i16_data.end());
                tensor->setTensorValueInt32(i32_data.size(), i32_data.data());
            }
            break;
            case DType_INT32: {
                std::vector<int32_t> i32_data;
                TosaSerializationHandler::ConvertU8toI32(data, tensor->getElementCount(), i32_data);
                tensor->setTensorValueInt32(i32_data.size(), i32_data.data());
            }
            break;
            case DType_INT48: {
                std::vector<int64_t> i64_data;
                TosaSerializationHandler::ConvertU8toI48(data, tensor->getElementCount(), i64_data);
                tensor->setTensorValueInt64(i64_data.size(), i64_data.data());
            }
            break;
            case DType_SHAPE: {
                std::vector<int64_t> i64_data;
                TosaSerializationHandler::ConvertU8toI64(data, tensor->getElementCount(), i64_data);
                tensor->setTensorValueInt64(i64_data.size(), i64_data.data());
            }
            break;
            case DType_FP16: {
                std::vector<half_float::half> f16_data;
                TosaSerializationHandler::ConvertU8toF16(data, tensor->getElementCount(), f16_data);
                if (tensor->getDtype() == TOSA_REF_TYPE_FP64)
                {
                    std::vector<double> f64_data(f16_data.begin(), f16_data.end());
                    tensor->setTensorValueDouble(f64_data.size(), f64_data.data());
                }
                else
                {
                    std::vector<float> f32_data(f16_data.begin(), f16_data.end());
                    tensor->setTensorValueFloat(f32_data.size(), f32_data.data());
                }
            }
            break;
            case DType_BF16: {
                std::vector<bf16> bf16_data;
                TosaSerializationHandler::ConvertU8toBF16(data, tensor->getElementCount(), bf16_data);
                if (tensor->getDtype() == TOSA_REF_TYPE_FP64)
                {
                    std::vector<double> f64_data;
                    for (auto f : bf16_data)
                    {
                        f64_data.push_back(static_cast<double>(f));
                    }
                    tensor->setTensorValueDouble(f64_data.size(), f64_data.data());
                }
                else
                {
                    std::vector<float> f32_data;
                    for (auto f : bf16_data)
                    {
                        f32_data.push_back(static_cast<float>(f));
                    }
                    tensor->setTensorValueFloat(f32_data.size(), f32_data.data());
                }
            }
            break;
            case DType_FP8E4M3: {
                std::vector<fp8e4m3> f8_data;
                TosaSerializationHandler::ConvertU8toFP8E4M3(data, tensor->getElementCount(), f8_data);
                if (tensor->getDtype() == TOSA_REF_TYPE_FP64)
                {
                    std::vector<double> f64_data;
                    for (auto f : f8_data)
                    {
                        f64_data.push_back(static_cast<double>(f));
                    }
                    tensor->setTensorValueDouble(f64_data.size(), f64_data.data());
                }
                else
                {
                    std::vector<float> f32_data;
                    for (auto f : f8_data)
                    {
                        f32_data.push_back(static_cast<float>(f));
                    }
                    tensor->setTensorValueFloat(f32_data.size(), f32_data.data());
                }
            }
            break;
            case DType_FP8E5M2: {
                std::vector<fp8e5m2> f8_data;
                TosaSerializationHandler::ConvertU8toFP8E5M2(data, tensor->getElementCount(), f8_data);
                if (tensor->getDtype() == TOSA_REF_TYPE_FP64)
                {
                    std::vector<double> f64_data;
                    for (auto f : f8_data)
                    {
                        f64_data.push_back(static_cast<double>(f));
                    }
                    tensor->setTensorValueDouble(f64_data.size(), f64_data.data());
                }
                else
                {
                    std::vector<float> f32_data;
                    for (auto f : f8_data)
                    {
                        f32_data.push_back(static_cast<float>(f));
                    }
                    tensor->setTensorValueFloat(f32_data.size(), f32_data.data());
                }
            }
            break;
            case DType_FP32: {
                std::vector<float> fp32_data;
                TosaSerializationHandler::ConvertU8toF32(data, tensor->getElementCount(), fp32_data);
                if (tensor->getDtype() == TOSA_REF_TYPE_FP64)
                {
                    std::vector<double> f64_data(fp32_data.begin(), fp32_data.end());
                    tensor->setTensorValueDouble(f64_data.size(), f64_data.data());
                }
                else
                {
                    tensor->setTensorValueFloat(fp32_data.size(), fp32_data.data());
                }
            }
            break;
            case DType_BOOL: {
                std::vector<bool> bool_data;
                TosaSerializationHandler::ConvertU8toBool(data, tensor->getElementCount(), bool_data);

                // std::vector<bool>::data() will return bit mask instead of array of bool array.
                // Need to translate manually.
                bool* bool_array = (bool*)calloc(bool_data.size(), sizeof(bool));
                for (size_t i = 0; i < bool_data.size(); i++)
                {
                    bool_array[i] = bool_data[i];
                }
                tensor->setTensorValueBool(bool_data.size(), bool_array);
            }
            break;
            default:
                SUBGRAPH_ERROR_IF(true, "SubgraphTraverser::initializeGraph(): Unsupported tensor type %s.",
                                  EnumNameDType(dtype));
        }
        tensor->setIsValid();
    }

    if (tensor->getIsValid())
    {
        // Push ready consumers to the next node list
        for (auto gn : tensor->getConsumers())
        {
            if (gn->hasAllInputsReady() && !gn->getOnNextNodeList() && !gn->getEvaluated())
            {
                addToNextNodeList(gn);
            }
        }
    }
    return 0;
}

int SubgraphTraverser::isFullyEvaluated() const
{
    return nextNodeList.empty();
}

GraphNode* SubgraphTraverser::getNextNode()
{
    GraphNode* nextNode = nextNodeList.front();
    ASSERT_MSG(nextNode, "SubgraphTraverser::getNextNode(): called with empty next node list");
    ASSERT_MSG(nextNode->getOnNextNodeList(),
               "SubgraphTraverser::getNextNode(): internal state error: node is not listed as being on next node list");

    nextNodeList.pop_front();

    nextNode->clearOnNextNodeList();
    return nextNode;
}

int SubgraphTraverser::addToNextNodeList(GraphNode* nextNode)
{
    ASSERT_MSG(nextNode, "SubgraphTraverser::addToNextNodeList(): called with no node");
    ASSERT_MSG(!nextNode->getOnNextNodeList(),
               "SubgraphTraverser::addToNextNodeList(): internal state error: node is already on next node list");

    nextNode->setOnNextNodeList();
    nextNodeList.push_back(nextNode);

    return 0;
}

int SubgraphTraverser::evaluateNextNode()
{
    if (isFullyEvaluated())
        return 0;

    GraphNode* currNode = getNextNode();

    DEBUG_INFO(GT, "Evaluating node_%03" PRIu64 ", %8s, output tensor=%s", currNode->getID(),
               EnumNamesOp()[currNode->getOp()], currNode->getOutputNames()[0].c_str());

    // Sanity check for never-ending loops
    if (currNode->getEvalCount() >= MAX_EVAL_COUNT && (currNode->getEvalCount() % MAX_EVAL_COUNT) == 0)
    {
        WARNING("SubgraphTraverser::evaluateNextNode(): Node %lu has been evaluated %d times.  Loop suspected.",
                currNode->getID(), currNode->getEvalCount());
    }

    for (auto tensor : currNode->getOutputs())
    {
        if (!tensor->is_allocated())
        {
            if (this->allocateTensor(tensor->getName()))
            {
                FATAL_ERROR("SubgraphTraverser::evaluateNextNode(): Failed to allocate Eigen tensor %s",
                            tensor->getName().c_str());
            }
        }
    }

    if (currNode->eval())
    {
        WARNING("SubgraphTraverser::evaluateNextNode(): Failed to evaluate node: %lu", currNode->getID());
        return 1;
    }

    currNode->setEvaluated();

    // free input tensor if all of its consumers have all of their outputs ready and it's not block's output
    for (auto tensor : currNode->getInputs())
    {
        bool in_use = false;

        auto tensor_check = findTensorByName(tensor->getName());
        if (tensor_check->getIsParentGraphOutput())
        {
            // if it's parent's block output tensor, we can't free it
            continue;
        }

        if (tensor->getIsVariable())
        {
            // if tensor is a Variable, we cannot free it
            continue;
        }

        for (auto node : tensor->getConsumers())
        {
            // If the node is inside a loop, the input tensor is still needed
            if (!node->hasAllOutputsReady())
            {
                in_use = true;
            }
        }

        for (auto name : block->GetOutputs())
        {
            if (name == tensor->getName())
            {
                in_use = true;
            }
        }

        if (!in_use)
        {
            tensor->deallocate();
        }
    }

    // Search the output tensors of this node to see if
    // there are now new ready nodes available from completing this node
    for (TosaReference::Tensor* tensor : currNode->getOutputs())
    {
        for (GraphNode* node : tensor->getConsumers())
        {
            if (!node->getOnNextNodeList() && node->hasAllInputsReady() && !node->getEvaluated())
            {
                addToNextNodeList(node);
            }
        }
    }

    if (DEBUG_ENABLED(DEBUG_VERB_HIGH, GT))
    {
        dumpNextNodeList(g_func_debug.func_debug_file);
    }

    if (g_func_config.dump_intermediates)
    {
        currNode->dumpNode(g_func_debug.func_debug_file);
        for (auto outs : currNode->getOutputs())
        {
            outs->dumpTensorParams(g_func_debug.func_debug_file);
            outs->dumpTensor(g_func_debug.func_debug_file);
            fprintf(g_func_debug.func_debug_file, "\n");
        }
    }

    return 0;
}

int SubgraphTraverser::dumpNextNodeList(FILE* out) const
{

    // Dump next node list
    fprintf(out, "Next node list\n");

    if (nextNodeList.empty())
    {
        fprintf(out, "<empty>\n");
    }

    for (auto gn : nextNodeList)
    {
        gn->dumpNode(out);
    }

    fprintf(out, "Done.\n");
    return 0;
}

int SubgraphTraverser::clearAllNodeMarkings()
{
    for (GraphNode* currNode : nodes)
    {
        currNode->clearNodeMarked();
    }

    return false;
}

int SubgraphTraverser::addTensor(const TosaSerializationTensor* ts)
{
    TosaReference::Tensor* tensor = nullptr;

    // variable tensors are shared: make new tensor only if not found
    if (ts->GetVariable())
    {
        tensor = getVariableTensorByName(ts->GetName());
    }

    if (!tensor)
    {
        DEBUG_INFO(GT, "Creating tensor %s", ts->GetName().c_str());
        tensor = TensorFactory::newTensor(ts->GetName(), ts->GetDtype(), ts->GetShape(),
                                          static_cast<uint32_t>(ts->GetShape().size()));

        SUBGRAPH_ERROR_IF(!tensor, "SubgraphTraverser::initializeGraph(): Unsupported tensor name=%s, type=%s, rank=%d",
                          ts->GetName().c_str(), EnumNameDType(ts->GetDtype()), (int)ts->GetShape().size());

        if (ts->GetVariable())
        {
            tensor->setIsVariable();
            registerVariableTensor(tensor);
        }
    }

    // Enforce no duplicate tensors/tensor names
    // O(N), but the number of tensors is small
    for (TosaReference::Tensor* currTensor : tensors)
    {
        if (tensor == currTensor || currTensor->getName() == tensor->getName())
        {
            FATAL_ERROR("SubgraphTraverser::addTensor(): Duplicate tensor or tensor name being added to graph: %s\n",
                        tensor->getName().c_str());
            return 1;
        }
    }

    tensors.push_back(tensor);

    if (tensor->getIsSubgraphInput())
    {
        inputTensors.push_back(tensor);
    }

    if (tensor->getIsSubgraphOutput())
    {
        outputTensors.push_back(tensor);
    }

    return 0;
}

int SubgraphTraverser::addShape(const TosaSerializationShape* ts)
{
    DEBUG_INFO(GT, "Creating tensor %s", ts->GetName().c_str());
    std::vector<int> shape        = { static_cast<int>(ts->GetRank()) };
    TosaReference::Tensor* tensor = TensorFactory::newTensor(ts->GetName(), DType_SHAPE, shape, 1);

    SUBGRAPH_ERROR_IF(!tensor, "SubgraphTraverser::initializeGraph(): Unsupported tensor name=%s, type=%s, rank=%d",
                      ts->GetName().c_str(), EnumNameDType(DType_SHAPE), 1);

    // Enforce no duplicate tensors/tensor names
    // O(N), but the number of tensors is small
    for (TosaReference::Tensor* currTensor : tensors)
    {
        if (tensor == currTensor || currTensor->getName() == tensor->getName())
        {
            FATAL_ERROR("SubgraphTraverser::addTensor(): Duplicate tensor or tensor name being added to graph: %s\n",
                        tensor->getName().c_str());
            return 1;
        }
    }

    tensors.push_back(tensor);

    return 0;
}

int SubgraphTraverser::addNode(GraphNode* newNode)
{
    // Enforce no duplicate nodes
    for (GraphNode* currNode : nodes)
    {
        if (currNode == newNode)
        {
            FATAL_ERROR("SubgraphTraverser::addNode(): duplicate node being added to graph");
            return 1;
        }
    }

    nodes.push_back(newNode);

    return 0;
}

TosaReference::Tensor* SubgraphTraverser::findTensorByName(const std::string& name) const
{
    TosaReference::Tensor* res_tensor = nullptr;

    for (TosaReference::Tensor* currTensor : tensors)
    {
        if (currTensor->getName() == name)
        {
            res_tensor = currTensor;
            return res_tensor;
        }
    }

    if (parent_sgt)
    {
        for (TosaReference::Tensor* currTensor : parent_sgt->tensors)
        {
            if (currTensor->getName() == name)
            {
                res_tensor = currTensor;
                res_tensor->setIsParentGraphOutput();
            }
        }
    }

    if (!res_tensor)
    {
        WARNING("SubgraphTraverser::findTensorByName(): Unable to find tensor with name: %s\n", name.c_str());
        return nullptr;
    }
    return res_tensor;
}

int SubgraphTraverser::linkTensorsAndNodes()
{
    // Nodes have a list of input/output tensor names
    // For each node, read this list, link up the tensors with their inputs/outputs
    for (GraphNode* currNode : nodes)
    {
        // Link inputs/consuming nodes
        for (std::string& name : currNode->getInputNames())
        {
            TosaReference::Tensor* t = findTensorByName(name);
            SUBGRAPH_ERROR_IF(!t,
                              "SubgraphTraverser::linkTensorsAndNodes(): Cannot find tensor %s in node %" PRIu64 "\n",
                              name.c_str(), currNode->getID());
            SUBGRAPH_ERROR_IF(currNode->addInputTensor(t),
                              "SubgraphTraverser::linkTensorsAndNodes(): cannot link tensor %s to node %" PRIu64 "\n ",
                              name.c_str(), currNode->getID());
            SUBGRAPH_ERROR_IF(t->addConsumer(currNode),
                              "SubgraphTraverser::linkTensorsAndNodes(): cannot link consumer node %" PRIu64
                              " to tensor %s\n",
                              currNode->getID(), name.c_str());
        }

        // Link outputs/producing nodes
        for (std::string& name : currNode->getOutputNames())
        {
            TosaReference::Tensor* t = findTensorByName(name);
            SUBGRAPH_ERROR_IF(!t,
                              "SubgraphTraverser::linkTensorsAndNodes(): Cannot find tensor %s in node %" PRIu64 "\n",
                              name.c_str(), currNode->getID());
            SUBGRAPH_ERROR_IF(currNode->addOutputTensor(t),
                              "SubgraphTraverser::linkTensorsAndNodes(): cannot link tensor %s to node %" PRIu64 "\n",
                              name.c_str(), currNode->getID());

            SUBGRAPH_ERROR_IF(t->setProducer(currNode),
                              "SubgraphTraverser::linkTensorsAndNodes(): cannot link producer node %" PRIu64
                              " to tensor tensor %s\n",
                              currNode->getID(), name.c_str());
        }
    }

    return 0;
}

int SubgraphTraverser::validateGraph()
{
    // Need to make sure that:
    //   - each tensor is actually used
    //   - input and output tesnsors truly are just input and just output
    // Graph building already determined that each node has found its input/output tensors

    for (TosaReference::Tensor* currTensor : tensors)
    {
        // It's okay for block input tensor not being consumed by operators.
        // This is common in control flow op execution.
        if (!currTensor->getIsSubgraphInput())
        {
            if (!currTensor->getProducer() && currTensor->getConsumers().empty())
            {
                WARNING("SubgraphTraverser::validateGraph(): TosaReference::Tensor %s has no producers or consumers\n",
                        currTensor->getName().c_str());
                return 1;
            }
        }
    }

    for (GraphNode* currNode : nodes)
    {
        SUBGRAPH_ERROR_IF(currNode->checkTensorAttributes(),
                          "SubgraphTraverser::validateGraph(): TosaReference::Tensor attribute check failed");
    }

    if (outputTensors.size() <= 0)
    {
        DEBUG_MED(GT, "Graph output tensor empty");
        return 0;
    }

    return 0;
}

int SubgraphTraverser::dumpGraph(FILE* out) const
{
    int i = 0;

    fprintf(out, "Full graph dump:\n");
    for (GraphNode* currNode : nodes)
    {
        fprintf(out, "Node [%d]: ", i++);
        currNode->dumpNode(out);
    }

    return 0;
}

int SubgraphTraverser::evaluateAll()
{
    // evaluation loop
    while (!isFullyEvaluated())
    {
        if (evaluateNextNode())
        {
            return 1;
        }
    }

    return 0;
}

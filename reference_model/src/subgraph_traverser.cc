
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

#include "subgraph_traverser.h"
#include "arith_util.h"
#include "tosa_model_types.h"

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
    return inputTensors.size();
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
    return outputTensors.size();
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

int SubgraphTraverser::initializeGraph()
{
    int idx = 0;

    std::vector<TosaSerializationTensor*> ser_tensor_vec;
    // Get all the serialized tensors from TosaSerializationHandler.
    if (tsh)
    {
        for (auto region : tsh->GetRegions())
        {
            for (auto block : region->GetBlocks())
            {
                for (auto ser_tensor : block->GetTensors())
                {
                    ser_tensor_vec.push_back(ser_tensor);
                }
            }
        }
    }
    else
    {
        for (auto ser_tensor : block->GetTensors())
        {
            ser_tensor_vec.push_back(ser_tensor);
        }
    }

    std::vector<GraphNode*> non_const_node_vec;
    for (auto op : block->GetOperators())
    {
        // translated TosaSerializationOperator to GraphNode
        TOSA_REF_TYPE input_dtype  = TOSA_REF_TYPE_UNKNOWN;
        TOSA_REF_TYPE output_dtype = TOSA_REF_TYPE_UNKNOWN;
        TOSA_REF_TYPE weight_dtype = TOSA_REF_TYPE_UNKNOWN;
        uint32_t input_rank        = 0;
        uint32_t output_rank       = 0;
        uint32_t weight_rank       = 0;
        int32_t input_index        = -1;
        int32_t weight_index       = -1;

        switch (op->GetOp())
        {
            case Op_CONV2D:
            case Op_CONV3D:
            case Op_DEPTHWISE_CONV2D:
            case Op_TRANSPOSE_CONV2D:
            case Op_FULLY_CONNECTED:
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

            std::string input_name                = op->GetInputTensorNames()[input_index];
            TosaSerializationTensor* input_tensor = nullptr;
            for (auto ser_tensor : ser_tensor_vec)
            {
                if (ser_tensor->GetName() == input_name)
                {
                    input_tensor = ser_tensor;
                }
            }

            SUBGRAPH_ERROR_IF(
                !input_tensor,
                "SubgraphTraverser::initializeGraph(): fail to get input tensor %s from TosaSerializationHandler",
                input_name.c_str());
            input_dtype = ConvertDType(input_tensor->GetDtype());
            input_rank  = input_tensor->GetShape().size();
        }

        if (weight_index != -1)
        {
            SUBGRAPH_ERROR_IF(
                (size_t)weight_index >= op->GetInputTensorNames().size(),
                "SubgraphTraverser::initializeGraph(): Op=%s, weight_index %d must be within [0, num_input - 1]",
                EnumNamesOp()[op->GetOp()], weight_index);
            std::string weight_name                = op->GetInputTensorNames()[weight_index];
            TosaSerializationTensor* weight_tensor = nullptr;
            for (auto ser_tensor : ser_tensor_vec)
            {
                if (ser_tensor->GetName() == weight_name)
                {
                    weight_tensor = ser_tensor;
                }
            }

            SUBGRAPH_ERROR_IF(
                !weight_tensor,
                "SubgraphTraverser::initializeGraph(): fail to get weight tensor %s from TosaSerializationHandler",
                weight_name.c_str());
            weight_dtype = ConvertDType(weight_tensor->GetDtype());
            weight_rank  = weight_tensor->GetShape().size();
        }

        SUBGRAPH_ERROR_IF(op->GetOutputTensorNames().size() == 0,
                          "SubgraphTraverser::initializeGraph(): Op=%s must have at least one output tensor.",
                          EnumNamesOp()[op->GetOp()]);
        std::string output_name                = op->GetOutputTensorNames()[0];
        TosaSerializationTensor* output_tensor = block->GetTensorByName(output_name);
        SUBGRAPH_ERROR_IF(
            !output_tensor,
            "SubgraphTraverser::initializeGraph(): fail to get output tensor %s from TosaSerializationHandler",
            output_name.c_str());
        output_dtype = ConvertDType(output_tensor->GetDtype());
        output_rank  = output_tensor->GetShape().size();

        DEBUG_INFO(GT, "Creating operator id_%03u, %8s, %lu input tensors, %lu output tensors", idx,
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

        idx++;
    }

    for (auto ts : block->GetTensors())
    {
        DEBUG_INFO(GT, "Creating tensor %s", ts->GetName().c_str());
        TosaReference::Tensor* tensor =
            TensorFactory::newTensor(ts->GetName(), ts->GetDtype(), ts->GetShape(), ts->GetShape().size());

        SUBGRAPH_ERROR_IF(!tensor, "SubgraphTraverser::initializeGraph(): Unsupported tensor name=%s, type=%s, rank=%d",
                          ts->GetName().c_str(), EnumNameDType(ts->GetDtype()), (int)ts->GetShape().size());

        addTensor(tensor);
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

int SubgraphTraverser::allocateTensor()
{
    for (auto ts : block->GetTensors())
    {
        // Bail out if tensor is used and any of its dimension is invalid.
        auto got = used_tensor_name_set.find(ts->GetName());
        if (got != used_tensor_name_set.end())
        {
            uint32_t elements = 1;
            for (auto& dim : ts->GetShape())
            {
                if (dim <= 0)
                {
                    DEBUG_INFO(GT, "Failed to allocate tensor %s with invalid dimension of %d", ts->GetName().c_str(),
                               dim);
                    this->setGraphStatus(GraphStatus::TOSA_UNPREDICTABLE);
                    return 1;
                }
                if (dim > static_cast<int32_t>(TOSA_MAX_TENSOR_SIZE / elements))
                {
                    // Size greather than maximum defined in spec
                    DEBUG_INFO(GT, "Tensor %s size is greater than allowed maximum", ts->GetName().c_str());
                    this->setGraphStatus(GraphStatus::TOSA_UNPREDICTABLE);
                    return 1;
                }
            }
        }

        TosaReference::Tensor* tensor = findTensorByName(ts->GetName());
        SUBGRAPH_ERROR_IF(!tensor, "SubgraphTraverser::allocateTensor(): can't find tensor %s.", ts->GetName().c_str());

        DEBUG_INFO(GT, "Allocating tensor %s", tensor->getName().c_str());
        if (tensor->allocate())
        {
            FATAL_ERROR("Failed to allocate tensor %s", tensor->getName().c_str());
        }

        if (!ts->GetData().empty())
        {
            DEBUG_INFO(GT, "Setting data for tensor %s", tensor->getName().c_str());
            auto serialization_dtype = ts->GetDtype();
            switch (serialization_dtype)
            {
                case DType_INT4: {
                    std::vector<int8_t> i4_data;
                    TosaSerializationHandler::ConvertU8toI4(ts->GetData(), tensor->getElementCount(), i4_data);
                    std::vector<int32_t> i32_data(i4_data.begin(), i4_data.end());
                    tensor->setTensorValueInt32(i32_data.size(), i32_data.data());
                }
                break;
                case DType_INT8: {
                    std::vector<int8_t> i8_data;
                    TosaSerializationHandler::ConvertU8toI8(ts->GetData(), tensor->getElementCount(), i8_data);
                    std::vector<int32_t> i32_data(i8_data.begin(), i8_data.end());
                    tensor->setTensorValueInt32(i32_data.size(), i32_data.data());
                }
                break;
                case DType_INT16: {
                    std::vector<int16_t> i16_data;
                    TosaSerializationHandler::ConvertU8toI16(ts->GetData(), tensor->getElementCount(), i16_data);
                    std::vector<int32_t> i32_data(i16_data.begin(), i16_data.end());
                    tensor->setTensorValueInt32(i32_data.size(), i32_data.data());
                }
                break;
                case DType_INT32: {
                    std::vector<int32_t> i32_data;
                    TosaSerializationHandler::ConvertU8toI32(ts->GetData(), tensor->getElementCount(), i32_data);
                    tensor->setTensorValueInt32(i32_data.size(), i32_data.data());
                }
                break;
                case DType_INT48: {
                    std::vector<int64_t> i64_data;
                    TosaSerializationHandler::ConvertU8toI48(ts->GetData(), tensor->getElementCount(), i64_data);
                    tensor->setTensorValueInt64(i64_data.size(), i64_data.data());
                }
                break;
                case DType_FP16: {
                    // Interpret f16 data as float
                    std::vector<float> f16_data;
                    TosaSerializationHandler::ConvertU8toF16(ts->GetData(), tensor->getElementCount(), f16_data);
                    if (tensor->getDtype() == TOSA_REF_TYPE_FP64)
                    {
                        std::vector<double> f64_data(f16_data.begin(), f16_data.end());
                        tensor->setTensorValueDouble(f64_data.size(), f64_data.data());
                    }
                    else
                    {
                        tensor->setTensorValueFloat(f16_data.size(), f16_data.data());
                    }
                }
                break;
                case DType_BF16: {
                    std::vector<float> fp32_data;
                    TosaSerializationHandler::ConvertU8toF32(ts->GetData(), tensor->getElementCount(), fp32_data);
                    // Ensure valid bfloat16 stored in each float
                    for (auto f : fp32_data)
                        ASSERT_MSG(checkValidBFloat(f), "Float value %f not valid bfloat16", f);
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
                case DType_FP32: {
                    std::vector<float> fp32_data;
                    TosaSerializationHandler::ConvertU8toF32(ts->GetData(), tensor->getElementCount(), fp32_data);
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
                    TosaSerializationHandler::ConvertU8toBool(ts->GetData(), tensor->getElementCount(), bool_data);

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
                                      EnumNameDType(ts->GetDtype()));
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

    DEBUG_INFO(GT, "Evaluating node_%03lu, %8s, output tensor=%s", currNode->getID(), EnumNamesOp()[currNode->getOp()],
               currNode->getOutputNames()[0].c_str());

    // Sanity check for never-ending loops
    if (currNode->getEvalCount() >= MAX_EVAL_COUNT && (currNode->getEvalCount() % MAX_EVAL_COUNT) == 0)
    {
        WARNING("SubgraphTraverser::evaluateNextNode(): Node %lu has been evaluated %d times.  Loop suspected.",
                currNode->getID(), currNode->getEvalCount());
    }

    for (auto tensor : currNode->getOutputs())
    {
        if (!tensor->is_allocated())
            if (tensor->allocate())
            {
                FATAL_ERROR("SubgraphTraverser::evaluateNextNode(): Failed to allocate Eigen tensor %s",
                            tensor->getName().c_str());
            }
    }

    if (currNode->eval())
    {
        WARNING("SubgraphTraverser::evaluateNextNode(): Failed to evaluate node: %lu", currNode->getID());
        return 1;
    }

    // free input tensor if all of its consumers have all of their outputs ready and it's not block's output
    if (!currNode->getInMainBlock())
    {    // we don't free it if the node is in main block and has nested blocks
        for (auto tensor : currNode->getInputs())
        {
            bool in_use = false;

            auto tensor_check = findTensorByName(tensor->getName());
            if (tensor_check->getIsParentGraphOutput())
            {
                // if it's parent's block output tensor, we can't free it
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
    }
    // Search the output tensors of this node to see if
    // there are now new ready nodes available from completing this node
    for (TosaReference::Tensor* tensor : currNode->getOutputs())
    {
        for (GraphNode* node : tensor->getConsumers())
        {
            if (!node->getOnNextNodeList() && node->hasAllInputsReady())
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

int SubgraphTraverser::addTensor(TosaReference::Tensor* tensor)
{
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
int SubgraphTraverser::addNode(GraphNode* newNode)
{
    // Enforce no duplicate nodes
    for (GraphNode* currNode : nodes)
    {
        if (currNode == newNode)
        {
            FATAL_ERROR("SubgraphTraverser::addTensor(): duplicate node being added to graph");
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
            SUBGRAPH_ERROR_IF(!t, "SubgraphTraverser::linkTensorsAndNodes(): Cannot find tensor %s in node %lu\n",
                              name.c_str(), currNode->getID());
            SUBGRAPH_ERROR_IF(currNode->addInputTensor(t),
                              "SubgraphTraverser::linkTensorsAndNodes(): cannot link tensor %s to node %lu\n",
                              name.c_str(), currNode->getID());
            SUBGRAPH_ERROR_IF(t->addConsumer(currNode),
                              "SubgraphTraverser::linkTensorsAndNodes(): cannot link consumer node %lu to tensor %s\n",
                              currNode->getID(), name.c_str());
        }

        // Link outputs/producing nodes
        for (std::string& name : currNode->getOutputNames())
        {
            TosaReference::Tensor* t = findTensorByName(name);
            SUBGRAPH_ERROR_IF(!t, "SubgraphTraverser::linkTensorsAndNodes(): Cannot find tensor %s in node %lu\n",
                              name.c_str(), currNode->getID());
            SUBGRAPH_ERROR_IF(currNode->addOutputTensor(t),
                              "SubgraphTraverser::linkTensorsAndNodes(): cannot link tensor %s to node %lu\n",
                              name.c_str(), currNode->getID());

            SUBGRAPH_ERROR_IF(
                t->setProducer(currNode),
                "SubgraphTraverser::linkTensorsAndNodes(): cannot link producer node %lu to tensor tensor %s\n",
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

        if (g_func_config.tosa_profile == 0)
        {
            TOSA_REF_TYPE dtype = currTensor->getDtype();

            // Float-point disallowed
            if (dtype == TOSA_REF_TYPE_FP32 || dtype == TOSA_REF_TYPE_FP16)
            {
                WARNING("SubgraphTraverser::validateGraph(): TOSA Base Inference profile selected: All floating point "
                        "disabled, but %s tensor %s found\n",
                        EnumNameTOSAREFTYPE(dtype), currTensor->getName().c_str());
                return 1;
            }
        }
        else if (g_func_config.tosa_profile == 1 || g_func_config.tosa_profile == 2)
        {
            // Do nothing. All FP types allowed
            // Currently no implementation difference between Main Inference and Main Training modes
        }
        else
        {
            FATAL_ERROR("SubgraphTraverser::validateGraph(): TOSA profile not recognized: %d",
                        g_func_config.tosa_profile);
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


// Copyright (c) 2020, ARM Limited.
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

using namespace TosaReference;
using namespace Eigen;
using namespace tosa;

SubgraphTraverser::SubgraphTraverser(TosaSerializationBasicBlock* _block, TosaSerializationHandler* _tsh)
{
    graph_status = GraphStatus::TOSA_VALID;

    block = _block;
    tsh   = _tsh;

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
    for (auto op : block->GetOperators())
    {
        // translated TosaSerializationOperator to GraphNode
        DType input_dtype    = DType_UNKNOWN;
        DType output_dtype   = DType_UNKNOWN;
        DType weight_dtype   = DType_UNKNOWN;
        uint32_t input_rank  = 0;
        uint32_t output_rank = 0;
        uint32_t weight_rank = 0;
        int32_t input_index  = -1;
        int32_t weight_index = -1;

        switch (op->GetOp())
        {
            case Op_CONV2D:
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
            ASSERT_MSG((size_t)input_index < op->GetInputTensorNames().size(),
                       "Op=%s, input_index %d must be within [0, num_input - 1]", EnumNamesOp()[op->GetOp()],
                       input_index);

            std::string input_name                = op->GetInputTensorNames()[input_index];
            TosaSerializationTensor* input_tensor = block->GetTensorByName(input_name);
            ASSERT_MSG(input_tensor, "SubgraphTraverser: fail to get input tensor %s from TosaSerializationHandler",
                       input_name.c_str());
            input_dtype = input_tensor->GetDtype();
            input_rank  = input_tensor->GetShape().size();
        }

        if (weight_index != -1)
        {
            ASSERT_MSG((size_t)weight_index < op->GetInputTensorNames().size(),
                       "Op=%s, weight_index %d must be within [0, num_input - 1]", EnumNamesOp()[op->GetOp()],
                       weight_index);
            std::string weight_name                = op->GetInputTensorNames()[weight_index];
            TosaSerializationTensor* weight_tensor = block->GetTensorByName(weight_name);
            ASSERT_MSG(weight_tensor, "SubgraphTraverser: fail to get weight tensor %s from TosaSerializationHandler",
                       weight_name.c_str());
            weight_dtype = weight_tensor->GetDtype();
            weight_rank  = weight_tensor->GetShape().size();
        }

        std::string output_name                = op->GetOutputTensorNames()[0];
        TosaSerializationTensor* output_tensor = block->GetTensorByName(output_name);
        ASSERT_MSG(output_tensor, "SubgraphTraverser: fail to get output tensor %s from TosaSerializationHandler",
                   output_name.c_str());
        output_dtype = output_tensor->GetDtype();
        output_rank  = output_tensor->GetShape().size();

        DEBUG_INFO(GT, "Creating operator id_%03u, %8s, %lu input tensors, %lu output tensors", idx,
                   EnumNamesOp()[op->GetOp()], op->GetInputTensorNames().size(), op->GetOutputTensorNames().size());

        GraphNode* node = OpFactory::newOp(this, tsh, op->GetOp(), op->GetAttribute(), op->GetQInfo(), idx, input_dtype,
                                           input_rank, output_dtype, output_rank, weight_dtype, weight_rank);
        if (!node)
        {
            if (weight_index == -1)
            {
                fprintf(g_func_debug.func_debug_file,
                        "OpFactory could not allocate op %8s input=(%s rank %d) -> (%s rank %d)",
                        EnumNamesOp()[op->GetOp()], EnumNamesDType()[input_dtype], input_rank,
                        EnumNamesDType()[output_dtype], output_rank);
            }
            else
            {
                fprintf(g_func_debug.func_debug_file,
                        "OpFactory could not allocate op %8s input=(%s rank %d), weight=(%s rank %d) -> (%s rank %d)",
                        EnumNamesOp()[op->GetOp()], EnumNamesDType()[input_dtype], input_rank,
                        EnumNamesDType()[weight_dtype], weight_rank, EnumNamesDType()[output_dtype], output_rank);
            }

            for (auto& ts : op->GetInputTensorNames())
            {
                fprintf(g_func_debug.func_debug_file, "Input: %s\n", ts.c_str());
            }

            for (auto& ts : op->GetOutputTensorNames())
            {
                fprintf(g_func_debug.func_debug_file, "Output: %s\n", ts.c_str());
            }
            FATAL_ERROR("Unsupported operation type or rank.");
        }

        for (auto& name : op->GetInputTensorNames())
        {
            node->addInputName(name);
        }

        for (auto name : op->GetOutputTensorNames())
        {
            node->addOutputName(name);
        }

        addNode(node);

        // if node doesn't have any inputs (i.e. CONST)
        // it should be ready for evaluation
        if (op->GetInputTensorNames().empty() && !node->getOnNextNodeList())
        {
            addToNextNodeList(node);
        }

        idx++;
    }

    for (auto ts : block->GetTensors())
    {
        // Bail out if any dimension is invalid.
        for (auto& dim : ts->GetShape())
        {
            if (dim <= 0)
            {
                this->setGraphStatus(GraphStatus::TOSA_UNPREDICTABLE);
                return 1;
            }
        }

        DEBUG_INFO(GT, "Creating tensor %s", ts->GetName().c_str());
        TosaReference::Tensor* tensor =
            TensorFactory::newTensor(ts->GetName(), ts->GetDtype(), ts->GetShape(), ts->GetShape().size());

        if (!ts->GetData().empty())
        {
            if (tensor->allocate())
            {
                SIMPLE_FATAL_ERROR("Failed to allocate tensor %s", tensor->getName().c_str());
            }

            switch (ts->GetDtype())
            {
                case DType_INT4:
                {
                    std::vector<int8_t> i4_data;
                    TosaSerializationHandler::ConvertU8toI4(ts->GetData(), tensor->getElementCount(), i4_data);
                    std::vector<int32_t> i32_data(i4_data.begin(), i4_data.end());
                    tensor->setTensorValueInt32(i32_data.size(), i32_data.data());
                }
                break;
                case DType_INT8:
                {
                    std::vector<int8_t> i8_data;
                    TosaSerializationHandler::ConvertU8toI8(ts->GetData(), tensor->getElementCount(), i8_data);
                    std::vector<int32_t> i32_data(i8_data.begin(), i8_data.end());
                    tensor->setTensorValueInt32(i32_data.size(), i32_data.data());
                }
                break;
                case DType_INT16:
                {
                    std::vector<int16_t> i16_data;
                    TosaSerializationHandler::ConvertU8toI16(ts->GetData(), tensor->getElementCount(), i16_data);
                    std::vector<int32_t> i32_data(i16_data.begin(), i16_data.end());
                    tensor->setTensorValueInt32(i32_data.size(), i32_data.data());
                }
                break;
                case DType_INT32:
                {
                    std::vector<int32_t> i32_data;
                    TosaSerializationHandler::ConvertU8toI32(ts->GetData(), tensor->getElementCount(), i32_data);
                    tensor->setTensorValueInt32(i32_data.size(), i32_data.data());
                }
                break;
                case DType_INT48:
                {
                    std::vector<int64_t> i64_data;
                    TosaSerializationHandler::ConvertU8toI48(ts->GetData(), tensor->getElementCount(), i64_data);
                    tensor->setTensorValueInt64(i64_data.size(), i64_data.data());
                }
                break;
                case DType_FLOAT:
                {
                    std::vector<float> fp32_data;
                    TosaSerializationHandler::ConvertU8toF32(ts->GetData(), tensor->getElementCount(), fp32_data);
                    tensor->setTensorValueFloat(fp32_data.size(), fp32_data.data());
                }
                break;
                case DType_BOOL:
                {
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
                    FATAL_ERROR("Unsupported tensor type %s.", EnumNamesDType()[ts->GetDtype()]);
            }
        }

        // update this->tensors
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
            FATAL_ERROR("loadGraphJson: Failed to find input tensor by name %s", input_name.c_str());
        }
    }

    DEBUG_INFO(GT, "Enumerating block %s graph outputs", block->GetName().c_str());
    for (auto& output_name : block->GetOutputs())
    {
        TosaReference::Tensor* tensor = findTensorByName(output_name);
        DEBUG_INFO(GT, "output tensor name=%s\n", output_name.c_str());
        if (tensor)
        {
            tensor->setIsSubgraphOutput();
            outputTensors.push_back(tensor);
        }
        else
        {
            FATAL_ERROR("loadGraphJson: Failed to find output tensor by name %s", output_name.c_str());
        }
    }

    if (DEBUG_ENABLED(DEBUG_VERB_HIGH, GT))
    {
        dumpNextNodeList(g_func_debug.func_debug_file);
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
        WARNING("Node %lu has been evaluated %d times.  Loop suspected.", currNode->getID(), currNode->getEvalCount());
    }

    for (auto tensor : currNode->getOutputs())
    {
        if (!tensor->is_allocated())
            if (tensor->allocate())
            {
                FATAL_ERROR("Failed to allocate Eigen tensor %s", tensor->getName().c_str());
            }
    }

    if (currNode->eval())
    {
        WARNING("Failed to evaluate node: %lu", currNode->getID());
        return 1;
    }

    // free input tensor if all of its consumers have all of their outputs ready and it's not block's output
    for (auto tensor : currNode->getInputs())
    {
        bool in_use = false;
        for (auto node : tensor->getConsumers())
        {
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
            FATAL_ERROR("Error: Duplicate tensor or tensor name being added to graph: %s\n", tensor->getName().c_str());
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
            FATAL_ERROR("Error: duplicate node being added to graph");
            return 1;
        }
    }

    nodes.push_back(newNode);

    return 0;
}

TosaReference::Tensor* SubgraphTraverser::findTensorByName(const std::string& name) const
{
    for (TosaReference::Tensor* currTensor : tensors)
    {
        if (currTensor->getName() == name)
        {
            return currTensor;
        }
    }

    WARNING("Unable to find tensor with name: %s\n", name.c_str());

    return nullptr;
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
            if (!t)
            {
                FATAL_ERROR("linkTensorsAndNodes: Cannot find tensor %s in node %lu\n", name.c_str(),
                            currNode->getID());
                return 1;
            }

            if (currNode->addInputTensor(t))
            {
                FATAL_ERROR("linkTensorsAndNodes: cannot link tensor %s to node %lu\n", name.c_str(),
                            currNode->getID());
                return 1;
            }

            if (t->addConsumer(currNode))
            {
                FATAL_ERROR("linkTensorsAndNodes: cannot link consumer node %lu to tensor %s\n", currNode->getID(),
                            name.c_str());
                return 1;
            }
        }

        // Link outputs/producing nodes
        for (std::string& name : currNode->getOutputNames())
        {
            TosaReference::Tensor* t = findTensorByName(name);
            if (!t)
            {
                FATAL_ERROR("linkTensorsAndNodes: Cannot find tensor %s in node %lu\n", name.c_str(),
                            currNode->getID());
                return 1;
            }

            if (currNode->addOutputTensor(t))
            {
                FATAL_ERROR("linkTensorsAndNodes: cannot link tensor %s to node %lu\n", name.c_str(),
                            currNode->getID());
                return 1;
            }

            if (t->setProducer(currNode))
            {
                FATAL_ERROR("linkTensorsAndNodes: cannot link producer node %lu to tensor tensor %s\n",
                            currNode->getID(), name.c_str());
                return 1;
            }
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
                WARNING("Graph inconsistency: TosaReference::Tensor %s has no producers or consumers\n",
                        currTensor->getName().c_str());
                return 1;
            }
        }

        if (g_func_config.tosa_profile == 0)
        {
            DType dtype = currTensor->getDtype();

            // Float-point disallowed
            if (dtype == DType_FLOAT)
            {
                WARNING("TOSA Base Inference profile selected: All floating point disabled, but %s tensor %s found\n",
                        EnumNamesDType()[dtype], currTensor->getName().c_str());
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
            FATAL_ERROR("TOSA profile not recognized: %d", g_func_config.tosa_profile);
        }
    }

    for (GraphNode* currNode : nodes)
    {
        if (currNode->checkTensorAttributes())
        {
            WARNING("TosaReference::Tensor attribute check failed");
            return 1;
        }
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

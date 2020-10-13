
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
    char tensor_fullname[1000];
    int idx = 0;
    for (auto op : block->GetOperators())
    {
        // translated TosaSerializationOperator to GraphNode
        DType in_dtype = DType_UNKNOWN, out_dtype = DType_UNKNOWN, weight_dtype = DType_UNKNOWN;
        uint32_t in_rank = 0, out_rank = 0, weight_rank = 0;
        for (auto name : op->GetInputTensorNames())
        {

            TosaSerializationTensor* ts = block->GetTensorByName(name);
            ASSERT_MSG(ts, "SubgraphTraverser: fail to get tensor %s from TosaSerializationHandler", name.c_str());

            if (ts->HasUsage(Usage_WEIGHT))
            {
                weight_dtype = ts->GetDtype();
                weight_rank  = ts->GetShape().size();
            }
            else if (ts->HasUsage(Usage_INDEX))
            {
                // do nothing, but this will prevent tensor's dtype/rank being wrongly used as template argument when initializing this op
            }
            else if (ts->HasUsage(Usage_ACTIVATION))
            {
                if (ts->GetShape().size() >= in_rank)
                {
                    in_dtype = ts->GetDtype();
                    in_rank  = ts->GetShape().size();
                }
            }
        }

        for (auto name : op->GetOutputTensorNames())
        {

            TosaSerializationTensor* ts = block->GetTensorByName(name);
            ASSERT_MSG(ts, "SubgraphTraverser: fail to get tensor %s from TosaSerializationHandler", name.c_str());

            out_dtype = ts->GetDtype();
            out_rank  = ts->GetShape().size();
        }

        DEBUG_INFO(GT, "Creating operator id_%03u, %8s, %lu input tensors, %lu output tensors", idx,
                   EnumNamesOp()[op->GetOp()], op->GetInputTensorNames().size(), op->GetOutputTensorNames().size());

        GraphNode* cn = OpFactory::newOp(tsh, op->GetOp(), op->GetAttribute(), op->GetQInfo(), idx, in_dtype, in_rank,
                                         out_dtype, out_rank, weight_dtype, weight_rank);
        if (!cn)
        {
            if (weight_dtype == DType_UNKNOWN && weight_rank == 0)
            {
                fprintf(g_func_debug.func_debug_file,
                        "OpFactory could not allocate op %8s input=(%s rank %d) -> (%s rank %d)",
                        EnumNamesOp()[op->GetOp()], EnumNamesDType()[in_dtype], in_rank, EnumNamesDType()[out_dtype],
                        out_rank);
            }
            else
            {
                fprintf(g_func_debug.func_debug_file,
                        "OpFactory could not allocate op %8s input=(%s rank %d), weight=(%s rank %d) -> (%s rank %d)",
                        EnumNamesOp()[op->GetOp()], EnumNamesDType()[in_dtype], in_rank, EnumNamesDType()[weight_dtype],
                        weight_rank, EnumNamesDType()[out_dtype], out_rank);
            }

            for (auto ts : op->GetInputTensors())
            {
                fprintf(g_func_debug.func_debug_file, "Input: %s\n", ts->GetName().c_str());
            }

            for (auto ts : op->GetOutputTensors())
            {
                fprintf(g_func_debug.func_debug_file, "Output: %s\n", ts->GetName().c_str());
            }
            FATAL_ERROR("Unsupported operation type or rank.");
        }

        for (auto name : op->GetInputTensorNames())
        {
            cn->addInputName(name);
        }

        for (auto name : op->GetOutputTensorNames())
        {
            cn->addOutputName(name);
        }

        addNode(cn);

        // if node doesn't have any inputs (i.e. CONST)
        // it should be ready for evaluation
        if (op->GetInputTensorNames().empty() && !cn->getOnNextNodeList())
        {
            addToNextNodeList(cn);
        }

        idx++;
    }

    for (auto ts : block->GetTensors())
    {

        bool is_const = false;
        if (ts->HasUsage(Usage_WEIGHT))
        {
            is_const = true;
        }

        DEBUG_INFO(GT, "Creating tensor %s", ts->GetName().c_str());
        TosaReference::Tensor* ct =
            TensorFactory::newTensor(ts->GetName(), ts->GetDtype(), ts->GetUsage(), ts->GetFormat(), ts->GetShape(),
                                     is_const, ts->GetShape().size());

        if (ts->GetNpyFilePtr())
        {
            if (ct->allocate())
            {
                FATAL_ERROR("Fail to allocate Eigen tensor %s", ct->getName().c_str());
            }

            bzero(tensor_fullname, sizeof(tensor_fullname));
            snprintf(tensor_fullname, sizeof(tensor_fullname), "%s/%s", g_func_config.subgraph_dir,
                     ts->GetNpyFilePtr()->c_str());
            if (ct->readFromNpyFile(tensor_fullname))
            {
                FATAL_ERROR("Cannot read input data into graph tensor %s from block %s", ct->getName().c_str(),
                            block->GetName().c_str());
            }
        }

        // update this->tensors
        addTensor(ct);
    }

    DEBUG_INFO(GT, "Enumerating block %s graph inputs", block->GetName().c_str());
    for (auto& input_name : block->GetInputs())
    {
        TosaReference::Tensor* ct = findTensorByName(input_name);
        DEBUG_INFO(GT, "input tensor name=%s", input_name.c_str());
        if (ct)
        {
            ct->setIsSubgraphInput();
            inputTensors.push_back(ct);
        }
        else
        {
            FATAL_ERROR("loadGraphJson: Fail to find input tensor by name %s", input_name.c_str());
        }
    }

    DEBUG_INFO(GT, "Enumerating block %s graph outputs", block->GetName().c_str());
    for (auto& output_name : block->GetOutputs())
    {
        TosaReference::Tensor* ct = findTensorByName(output_name);
        DEBUG_INFO(GT, "output tensor name=%s\n", output_name.c_str());
        if (ct)
        {
            ct->setIsSubgraphOutput();
            outputTensors.push_back(ct);
        }
        else
        {
            FATAL_ERROR("loadGraphJson: Fail to find output tensor by name %s", output_name.c_str());
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

    for (auto ct : currNode->getOutputs())
    {
        if (!ct->is_allocated())
            if (ct->allocate())
            {
                FATAL_ERROR("Fail to allocate Eigen tensor %s", ct->getName().c_str());
            }
    }

    if (currNode->eval())
    {
        FATAL_ERROR("Error evaluating node: %lu\n", currNode->getID());
    }

    // free input tensor if all of its consumers have all of their outputs ready and it's not block's output
    for (auto ct : currNode->getInputs())
    {
        bool in_use = false;
        for (auto cn : ct->getConsumers())
        {
            if (!cn->hasAllOutputsReady())
            {
                in_use = true;
            }
        }
        for (auto name : block->GetOutputs())
        {
            if (name == ct->getName())
            {
                in_use = true;
            }
        }
        if (!in_use)
        {
            ct->deallocate();
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

int SubgraphTraverser::addTensor(TosaReference::Tensor* ct)
{
    // Enforce no duplicate tensors/tensor names
    // O(N), but the number of tensors is small
    for (TosaReference::Tensor* currTensor : tensors)
    {
        if (ct == currTensor || currTensor->getName() == ct->getName())
        {
            FATAL_ERROR("Error: Duplicate tensor or tensor name being added to graph: %s\n", ct->getName().c_str());
            return 1;
        }
    }

    tensors.push_back(ct);

    if (ct->getIsSubgraphInput())
    {
        inputTensors.push_back(ct);
    }

    if (ct->getIsSubgraphOutput())
    {
        outputTensors.push_back(ct);
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

        if (!currTensor->getProducer() && currTensor->getConsumers().empty())
        {
            WARNING("Graph inconsistency: TosaReference::Tensor %s has no producers or consumers\n",
                    currTensor->getName().c_str());
            return 1;
        }

        if (currTensor->getIsSubgraphInput())
        {
            if (currTensor->getProducer() && currTensor->getProducer()->getOp() != Op_PLACEHOLDER)
            {
                WARNING("Graph inconsistency: TosaReference::Tensor %s is a subgraph input and has a producer\n",
                        currTensor->getName().c_str());
                return 1;
            }
        }

        // comment this check out as this is possible when graph have multiple output
        // for example:
        //   %0 = add(%arg0, %arg1)
        //   %1 = mul(%arg0, %0)
        //   yields(%0, %1)
        //if (currTensor->getIsSubgraphOutput()) {
        //    if (!currTensor->getConsumers().empty()) {
        //        WARNING ("Graph inconsistency: TosaReference::Tensor %s is a subgraph output and has a consumer\n",
        //                     currTensor->getName().c_str());
        //        return 1;
        //    }
        //}

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


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

#include "graph_node.h"

using namespace TosaReference;
using namespace Eigen;
using namespace tosa;

GraphNode::GraphNode(SubgraphTraverser* parent_sgt_, const Op& nodeType_, const uint64_t id_)
{
    parent_sgt = parent_sgt_;
    nodeType   = nodeType_;
    nodeId     = id_;
    inputs.clear();
    outputs.clear();
    inputNames.clear();
    outputNames.clear();
    clearNodeMarked();
    evalCount = 0;
    clearOnNextNodeList();
    clearEvaluated();
    setRequiredOperands(-1, -1);
    setRequiredRank(-1);
    inMainBlock = false;
}

GraphNode::~GraphNode()
{}

int GraphNode::addInputName(std::string& name)
{
    inputNames.push_back(name);
    return 0;
}

int GraphNode::addOutputName(std::string& name)
{
    outputNames.push_back(name);
    return 0;
}

int GraphNode::addInputTensor(Tensor* tens)
{
    ASSERT_MSG(tens, "GraphNode::addInputTensor: no tensor provided");
    inputs.push_back(tens);
    return 0;
}

int GraphNode::addOutputTensor(Tensor* tens)
{
    ASSERT_MSG(tens, "GraphNode::addOutputTensor: no tensor provided");
    outputs.push_back(tens);
    return 0;
}

int GraphNode::checkTensorAttributes()
{
    // Placeholder
    return 0;
}

int GraphNode::eval()
{
    // Placeholder evaluation function
    evalCount++;

    // this should be set by derived op
    for (auto ct : getOutputs())
    {
        ct->setIsValid();
    }

    return 0;
}

int GraphNode::hasAllInputsReady() const
{
    for (size_t i = 0; i < inputs.size(); i++)
    {
        if (!inputs[i]->getIsValid())
            return false;
    }

    return true;
}

int GraphNode::hasAllOutputsReady() const
{
    for (size_t i = 0; i < outputs.size(); i++)
    {
        if (!outputs[i]->getIsValid())
            return false;
        if (outputs[i]->getIsVariable())
        {
            // when output is a variable tensor
            // isValid is not reliable indicator of this node having been evaluated
            return false;
        }
    }

    return true;
}

int GraphNode::dumpNode(FILE* out)
{
    int i;
    fprintf(out, "Node type: %s ID: %lu Eval Count: %d On next node list: %d Evaluated: %d Is marked: %d\n",
            EnumNamesOp()[nodeType], nodeId, evalCount, onNextNodeList, evaluated, isMarked);

    i = 0;
    for (Tensor* ins : inputs)
    {
        fprintf(out, "    Input[%d]  ", i++);
        ins->dumpTensorParams(out);
    }

    i = 0;
    for (Tensor* outs : outputs)
    {
        fprintf(out, "   Output[%d]  ", i++);
        outs->dumpTensorParams(out);
    }

    return 0;
}

int GraphNode::dumpNode(std::ostream& out)
{
    int i;

    out << "Node type: " << EnumNamesOp()[nodeType] << " ID: " << nodeId << " Eval count: " << evalCount
        << " On next node list: " << onNextNodeList << "Evaluated: " << evaluated << " Is marked: " << isMarked
        << std::endl;

    out << "  Inputs:";
    for (std::string& name : inputNames)
    {
        out << " " << name;
    }
    out << std::endl;

    i = 0;
    for (Tensor* ins : inputs)
    {
        out << "    Input[" << i++ << "]: ";
        ins->dumpTensorParams(out);
    }

    out << "  Outputs:";
    for (std::string& name : outputNames)
    {
        out << " " << name;
    }
    out << std::endl;

    i = 0;
    for (Tensor* outs : outputs)
    {
        out << "    Output[" << i++ << "]: ";
        outs->dumpTensorParams(out);
    }
    return 0;
}

int GraphNode::printNodeValidationError(const std::string& msg)
{
    std::cout << "Operator validation error: " << msg << std::endl;
    ;
    dumpNode(std::cout);

    return 0;
}

int GraphNode::validateRequiredOperands()
{
    if (requiredInputCount >= 0 && inputs.size() != (size_t)requiredInputCount)
    {
        printNodeValidationError(std::string(EnumNamesOp()[nodeType]) + " operator must have " +
                                 std::to_string(requiredInputCount) + " input(s)");
        return 1;
    }

    if (requiredOutputCount >= 0 && outputs.size() != (size_t)requiredOutputCount)
    {
        printNodeValidationError(std::string(EnumNamesOp()[nodeType]) + " operator output must have exactly " +
                                 std::to_string(requiredOutputCount) + " output(s)");
        return 1;
    }

    return 0;
}

int GraphNode::validateRequiredRank(const Tensor* t)
{
    if (requiredRankMin >= 0 && requiredRankMax >= 0)
    {
        std::string err_message = std::string(EnumNamesOp()[nodeType]) +
                                  " operand has illegal rank=" + std::to_string(t->getRank()) + " not in range [" +
                                  std::to_string(requiredRankMin) + "," + std::to_string(requiredRankMax) +
                                  "]. tensorName: " + t->getName();
        ERROR_IF(t->checkRequiredRank(requiredRankMin, requiredRankMax), "%s", err_message.c_str());

        return 0;
    }

    if (requiredRankMin >= 0)
    {
        std::string err_message = std::string(EnumNamesOp()[nodeType]) +
                                  " operand has illegal rank=" + std::to_string(t->getRank()) + " not equal to " +
                                  std::to_string(requiredRankMin) + ". tensorName: " + t->getName();
        ERROR_IF(t->checkRequiredRank(requiredRankMin), "%s", err_message.c_str());

        return 0;
    }

    return 0;
}

int GraphNode::idiv_check(int input1, int input2, int& result)
{
    ERROR_IF(input2 == 0, "idiv_check: input2 must not be zero");
    ERROR_IF(input1 % input2 != 0, "idiv_check: input1 must be a multiple of input2");

    result = input1 / input2;
    return 0;
}

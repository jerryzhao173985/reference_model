
// Copyright (c) 2020-2022, ARM Limited.
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

#ifndef GRAPH_NODE_H
#define GRAPH_NODE_H

#include "attribute.h"
#include "subgraph_traverser.h"
#include "tensor.h"
#include "tosa_generated.h"
#include <iostream>

#define DEF_INSTANTIATE_ONE_RANK_ONE_TYPE(OP, RANK, DTYPE) template class TosaReference::OP<RANK, DType_##DTYPE>;

#define DEF_INSTANTIATE_ONE_RANK_ONE_TYPE_ONE_ACCUM(OP, RANK, DTYPE, ACCUM_DTYPE)                                      \
    template class TosaReference::OP<RANK, DType_##DTYPE, DType_##ACCUM_DTYPE>;

#define DEF_INSTANTIATE_ONE_RANK_TWO_TYPE(OP, RANK, DTYPE1, DTYPE2)                                                    \
    template class TosaReference::OP<RANK, DType_##DTYPE1, DType_##DTYPE2>;

#define DEF_INSTANTIATE_TWO_RANK_ONE_TYPE(OP, RANK1, RANK2, DTYPE)                                                     \
    template class TosaReference::OP<RANK1, RANK2, DType_##DTYPE>;

#define DEF_INSTANTIATE_TWO_RANK_TWO_TYPE(OP, RANK1, RANK2, DTYPE1, DTYPE2)                                            \
    template class TosaReference::OP<RANK1, RANK2, DType_##DTYPE1, DType_##DTYPE2>;

#define DEF_INSTANTIATE_ONE_TYPE(OP, DTYPE) template class TosaReference::OP<DType_##DTYPE>;

#define DEF_INSTANTIATE_ONE_TYPE_ONE_ACCUM(OP, DTYPE, ACCUM_DTYPE)                                                     \
    template class TosaReference::OP<DType_##DTYPE, DType_##ACCUM_DTYPE>;

#define DEF_INSTANTIATE_TWO_TYPE(OP, DTYPE1, DTYPE2) template class TosaReference::OP<DType_##DTYPE1, DType_##DTYPE2>;

#define DEF_INSTANTIATE_TWO_TYPE_ONE_ACCUM(OP, DTYPE1, DTYPE2, ACCUM_DTYPE)                                            \
    template class TosaReference::OP<DType_##DTYPE1, DType_##DTYPE2, DType_##ACCUM_DTYPE>;

#define DEF_INSTANTIATE_THREE_TYPE(OP, DTYPE1, DTYPE2, OP_TYPE)                                                        \
    template class TosaReference::OP<DType_##DTYPE1, DType_##DTYPE2, OP_TYPE>;

#define DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OP, DTYPE)                                                           \
    DEF_INSTANTIATE_ONE_RANK_ONE_TYPE(OP, 0, DTYPE)                                                                    \
    DEF_INSTANTIATE_ONE_RANK_ONE_TYPE(OP, 1, DTYPE)                                                                    \
    DEF_INSTANTIATE_ONE_RANK_ONE_TYPE(OP, 2, DTYPE)                                                                    \
    DEF_INSTANTIATE_ONE_RANK_ONE_TYPE(OP, 3, DTYPE)                                                                    \
    DEF_INSTANTIATE_ONE_RANK_ONE_TYPE(OP, 4, DTYPE)                                                                    \
    DEF_INSTANTIATE_ONE_RANK_ONE_TYPE(OP, 5, DTYPE)                                                                    \
    DEF_INSTANTIATE_ONE_RANK_ONE_TYPE(OP, 6, DTYPE)

#define DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OP, DTYPE)                                                           \
    DEF_INSTANTIATE_ONE_RANK_ONE_TYPE(OP, 1, DTYPE)                                                                    \
    DEF_INSTANTIATE_ONE_RANK_ONE_TYPE(OP, 2, DTYPE)                                                                    \
    DEF_INSTANTIATE_ONE_RANK_ONE_TYPE(OP, 3, DTYPE)                                                                    \
    DEF_INSTANTIATE_ONE_RANK_ONE_TYPE(OP, 4, DTYPE)                                                                    \
    DEF_INSTANTIATE_ONE_RANK_ONE_TYPE(OP, 5, DTYPE)                                                                    \
    DEF_INSTANTIATE_ONE_RANK_ONE_TYPE(OP, 6, DTYPE)

#define DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE_ONE_ACCUM(OP, DTYPE, ACCUM_DTYPE)                                    \
    DEF_INSTANTIATE_ONE_RANK_ONE_TYPE_ONE_ACCUM(OP, 1, DTYPE, ACCUM_DTYPE)                                             \
    DEF_INSTANTIATE_ONE_RANK_ONE_TYPE_ONE_ACCUM(OP, 2, DTYPE, ACCUM_DTYPE)                                             \
    DEF_INSTANTIATE_ONE_RANK_ONE_TYPE_ONE_ACCUM(OP, 3, DTYPE, ACCUM_DTYPE)                                             \
    DEF_INSTANTIATE_ONE_RANK_ONE_TYPE_ONE_ACCUM(OP, 4, DTYPE, ACCUM_DTYPE)                                             \
    DEF_INSTANTIATE_ONE_RANK_ONE_TYPE_ONE_ACCUM(OP, 5, DTYPE, ACCUM_DTYPE)                                             \
    DEF_INSTANTIATE_ONE_RANK_ONE_TYPE_ONE_ACCUM(OP, 6, DTYPE, ACCUM_DTYPE)

#define DEF_INSTANTIATE_RANK0_6_ONE_RANK_TWO_TYPE(OP, DTYPE1, DTYPE2)                                                  \
    DEF_INSTANTIATE_ONE_RANK_TWO_TYPE(OP, 0, DTYPE1, DTYPE2)                                                           \
    DEF_INSTANTIATE_ONE_RANK_TWO_TYPE(OP, 1, DTYPE1, DTYPE2)                                                           \
    DEF_INSTANTIATE_ONE_RANK_TWO_TYPE(OP, 2, DTYPE1, DTYPE2)                                                           \
    DEF_INSTANTIATE_ONE_RANK_TWO_TYPE(OP, 3, DTYPE1, DTYPE2)                                                           \
    DEF_INSTANTIATE_ONE_RANK_TWO_TYPE(OP, 4, DTYPE1, DTYPE2)                                                           \
    DEF_INSTANTIATE_ONE_RANK_TWO_TYPE(OP, 5, DTYPE1, DTYPE2)                                                           \
    DEF_INSTANTIATE_ONE_RANK_TWO_TYPE(OP, 6, DTYPE1, DTYPE2)

#define DEF_INSTANTIATE_RESHAPE(OP, DTYPE)                                                                             \
    DEF_INSTANTIATE_TWO_RANK_ONE_TYPE(OP, 0, 0, DTYPE)                                                                 \
    DEF_INSTANTIATE_TWO_RANK_ONE_TYPE(OP, 0, 1, DTYPE)                                                                 \
    DEF_INSTANTIATE_TWO_RANK_ONE_TYPE(OP, 0, 2, DTYPE)                                                                 \
    DEF_INSTANTIATE_TWO_RANK_ONE_TYPE(OP, 0, 3, DTYPE)                                                                 \
    DEF_INSTANTIATE_TWO_RANK_ONE_TYPE(OP, 0, 4, DTYPE)                                                                 \
    DEF_INSTANTIATE_TWO_RANK_ONE_TYPE(OP, 0, 5, DTYPE)                                                                 \
    DEF_INSTANTIATE_TWO_RANK_ONE_TYPE(OP, 0, 6, DTYPE)                                                                 \
    DEF_INSTANTIATE_TWO_RANK_ONE_TYPE(OP, 1, 0, DTYPE)                                                                 \
    DEF_INSTANTIATE_TWO_RANK_ONE_TYPE(OP, 1, 1, DTYPE)                                                                 \
    DEF_INSTANTIATE_TWO_RANK_ONE_TYPE(OP, 1, 2, DTYPE)                                                                 \
    DEF_INSTANTIATE_TWO_RANK_ONE_TYPE(OP, 1, 3, DTYPE)                                                                 \
    DEF_INSTANTIATE_TWO_RANK_ONE_TYPE(OP, 1, 4, DTYPE)                                                                 \
    DEF_INSTANTIATE_TWO_RANK_ONE_TYPE(OP, 1, 5, DTYPE)                                                                 \
    DEF_INSTANTIATE_TWO_RANK_ONE_TYPE(OP, 1, 6, DTYPE)                                                                 \
    DEF_INSTANTIATE_TWO_RANK_ONE_TYPE(OP, 2, 0, DTYPE)                                                                 \
    DEF_INSTANTIATE_TWO_RANK_ONE_TYPE(OP, 2, 1, DTYPE)                                                                 \
    DEF_INSTANTIATE_TWO_RANK_ONE_TYPE(OP, 2, 2, DTYPE)                                                                 \
    DEF_INSTANTIATE_TWO_RANK_ONE_TYPE(OP, 2, 3, DTYPE)                                                                 \
    DEF_INSTANTIATE_TWO_RANK_ONE_TYPE(OP, 2, 4, DTYPE)                                                                 \
    DEF_INSTANTIATE_TWO_RANK_ONE_TYPE(OP, 2, 5, DTYPE)                                                                 \
    DEF_INSTANTIATE_TWO_RANK_ONE_TYPE(OP, 2, 6, DTYPE)                                                                 \
    DEF_INSTANTIATE_TWO_RANK_ONE_TYPE(OP, 3, 0, DTYPE)                                                                 \
    DEF_INSTANTIATE_TWO_RANK_ONE_TYPE(OP, 3, 1, DTYPE)                                                                 \
    DEF_INSTANTIATE_TWO_RANK_ONE_TYPE(OP, 3, 2, DTYPE)                                                                 \
    DEF_INSTANTIATE_TWO_RANK_ONE_TYPE(OP, 3, 3, DTYPE)                                                                 \
    DEF_INSTANTIATE_TWO_RANK_ONE_TYPE(OP, 3, 4, DTYPE)                                                                 \
    DEF_INSTANTIATE_TWO_RANK_ONE_TYPE(OP, 3, 5, DTYPE)                                                                 \
    DEF_INSTANTIATE_TWO_RANK_ONE_TYPE(OP, 3, 6, DTYPE)                                                                 \
    DEF_INSTANTIATE_TWO_RANK_ONE_TYPE(OP, 4, 0, DTYPE)                                                                 \
    DEF_INSTANTIATE_TWO_RANK_ONE_TYPE(OP, 4, 1, DTYPE)                                                                 \
    DEF_INSTANTIATE_TWO_RANK_ONE_TYPE(OP, 4, 2, DTYPE)                                                                 \
    DEF_INSTANTIATE_TWO_RANK_ONE_TYPE(OP, 4, 3, DTYPE)                                                                 \
    DEF_INSTANTIATE_TWO_RANK_ONE_TYPE(OP, 4, 4, DTYPE)                                                                 \
    DEF_INSTANTIATE_TWO_RANK_ONE_TYPE(OP, 4, 5, DTYPE)                                                                 \
    DEF_INSTANTIATE_TWO_RANK_ONE_TYPE(OP, 4, 6, DTYPE)                                                                 \
    DEF_INSTANTIATE_TWO_RANK_ONE_TYPE(OP, 5, 0, DTYPE)                                                                 \
    DEF_INSTANTIATE_TWO_RANK_ONE_TYPE(OP, 5, 1, DTYPE)                                                                 \
    DEF_INSTANTIATE_TWO_RANK_ONE_TYPE(OP, 5, 2, DTYPE)                                                                 \
    DEF_INSTANTIATE_TWO_RANK_ONE_TYPE(OP, 5, 3, DTYPE)                                                                 \
    DEF_INSTANTIATE_TWO_RANK_ONE_TYPE(OP, 5, 4, DTYPE)                                                                 \
    DEF_INSTANTIATE_TWO_RANK_ONE_TYPE(OP, 5, 5, DTYPE)                                                                 \
    DEF_INSTANTIATE_TWO_RANK_ONE_TYPE(OP, 5, 6, DTYPE)                                                                 \
    DEF_INSTANTIATE_TWO_RANK_ONE_TYPE(OP, 6, 0, DTYPE)                                                                 \
    DEF_INSTANTIATE_TWO_RANK_ONE_TYPE(OP, 6, 1, DTYPE)                                                                 \
    DEF_INSTANTIATE_TWO_RANK_ONE_TYPE(OP, 6, 2, DTYPE)                                                                 \
    DEF_INSTANTIATE_TWO_RANK_ONE_TYPE(OP, 6, 3, DTYPE)                                                                 \
    DEF_INSTANTIATE_TWO_RANK_ONE_TYPE(OP, 6, 4, DTYPE)                                                                 \
    DEF_INSTANTIATE_TWO_RANK_ONE_TYPE(OP, 6, 5, DTYPE)                                                                 \
    DEF_INSTANTIATE_TWO_RANK_ONE_TYPE(OP, 6, 6, DTYPE)

#define INIT_ATTRIBUTE(ATTRIBUTE_NAME)                                                                                 \
    if (auto p = dynamic_cast<Tosa##ATTRIBUTE_NAME##Attribute*>(attribute_))                                           \
    {                                                                                                                  \
        attribute = new Tosa##ATTRIBUTE_NAME##Attribute(p);                                                            \
        ASSERT_MEM(attribute);                                                                                         \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        FATAL_ERROR("Can't initialize Tosa" #ATTRIBUTE_NAME "Attribute");                                              \
    }

namespace TosaReference
{

class SubgraphTraverser;

// Nodes in the graph (e.g., tosa operators) are defined with this base
// class.
class GraphNode
{
public:
    GraphNode(SubgraphTraverser* parent_sgt_, const tosa::Op& nodeType_, const uint64_t id_);
    virtual ~GraphNode();

    int addInputName(std::string& name);
    int addOutputName(std::string& name);

    int addInputTensor(Tensor* tens);
    int addOutputTensor(Tensor* tens);

    // Validate that the input tensors match properly
    // in their types, attributes, rank, etc well enough to be
    // processed.
    //
    // This function should be pure virtual (eventually) in order to force
    // derivative operators to implement the check, but we'll initially
    // provide a default function so that GraphNode can be instantiated
    // directly for testing purposes.
    virtual int checkTensorAttributes();

    // Evalute the node/operator
    virtual int eval();

    int hasAllInputsReady() const;
    int hasAllOutputsReady() const;

    int dumpNode(FILE* out);
    int dumpNode(std::ostream& out);

    int setNodeMarked()
    {
        isMarked = true;
        return 0;
    }

    int getNodeMarked() const
    {
        return isMarked;
    }

    int clearNodeMarked()
    {
        isMarked = false;
        return 0;
    }

    int getEvalCount() const
    {
        return evalCount;
    }

    uint64_t getID() const
    {
        return nodeId;
    }

    std::vector<std::string>& getInputNames()
    {
        return inputNames;
    }

    std::vector<std::string>& getOutputNames()
    {
        return outputNames;
    }

    std::vector<Tensor*>& getOutputs()
    {
        return outputs;
    }

    std::vector<Tensor*>& getInputs()
    {
        return inputs;
    }

    int getOnNextNodeList() const
    {
        return onNextNodeList;
    }

    int setOnNextNodeList()
    {
        onNextNodeList = true;
        return 0;
    }

    int clearOnNextNodeList()
    {
        onNextNodeList = false;
        return 0;
    }

    tosa::Op getOp() const
    {
        return nodeType;
    }

    // Helper functions.
    int idiv_check(int input1, int input2, int& result);

protected:
    // Print out a node validation error
    int printNodeValidationError(const std::string& msg);

    int setRequiredOperands(const int in, const int out)
    {
        requiredInputCount  = in;
        requiredOutputCount = out;
        return 0;
    }

    int setRequiredRank(const int min, const int max = -1)
    {
        if (max == -1)
        {
            requiredRankMin = requiredRankMax = min;
        }
        else
        {
            requiredRankMin = min;
            requiredRankMax = max;
        }

        ASSERT_MSG(requiredRankMin <= requiredRankMax,
                   "GraphNode::setRequiredRank: requiredRankMin %d must be <= requiredRankMax %d", requiredRankMin,
                   requiredRankMax);

        return 0;
    }

    int validateRequiredOperands();
    int validateRequiredRank(const Tensor* t);

    // Parent SubgraphTraverser
    SubgraphTraverser* parent_sgt;

    // Description of the node type (e.g., CONST, CONV2D, etc...)
    tosa::Op nodeType;

    // A list of input tensor names
    std::vector<std::string> inputNames;

    // A list of the output tensor names
    std::vector<std::string> outputNames;

    // A list of the input tensors (after names have been matched up)
    std::vector<Tensor*> inputs;

    // A list of the output tensors (after names have been matched up)
    std::vector<Tensor*> outputs;

    // Unique node ID for debugging
    uint64_t nodeId;

    // Flag used for graph analysis
    int isMarked;

    // Number of times eval() has been called for this node
    int evalCount;

    // Flag indicating that this node is ready and is on the
    // next-node list.
    int onNextNodeList;

    // Required input/output tensor counts for node validation
    // -1 means any number is allowed
    int requiredInputCount;
    int requiredOutputCount;

    // Required rank ranges for input/output tensors
    // -1 means n/a
    int requiredRankMin;
    int requiredRankMax;
};

};    // namespace TosaReference

#endif

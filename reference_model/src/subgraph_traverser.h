
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

#ifndef SUBGRAPH_TRAVERSER_H
#define SUBGRAPH_TRAVERSER_H

#include "model_common.h"

#include "graph_node.h"
#include "ops/op_factory.h"
#include "tosa_serialization_handler.h"

namespace TosaReference
{

class SubgraphTraverser
{
public:
    SubgraphTraverser(TosaSerializationBasicBlock* block, TosaSerializationHandler* tsh);
    ~SubgraphTraverser();

    int initializeGraph();
    int isFullyEvaluated() const;
    int evaluateNextNode();
    int evaluateAll();

    int linkTensorsAndNodes();
    int validateGraph();

    int dumpGraph(FILE* out) const;
    int dumpNextNodeList(FILE* out) const;
    int clearAllNodeMarkings();

    int getNumInputTensors() const;
    Tensor* getInputTensor(const unsigned int idx) const;
    Tensor* getInputTensorByName(const std::string name) const;
    int getNumOutputTensors() const;
    Tensor* getOutputTensor(const unsigned int idx) const;
    Tensor* getOutputTensorByName(const std::string name) const;
    int addToNextNodeList(GraphNode*);

private:
    int addTensor(Tensor* ct);
    int addNode(GraphNode* cn);

    Tensor* findTensorByName(const std::string& name) const;

    GraphNode* getNextNode();

    // pointer to serialization library and corresponding basic block
    TosaSerializationBasicBlock* block;
    TosaSerializationHandler* tsh;

    // The definitive list of all tensors
    std::vector<Tensor*> tensors;

    // The subset of tensors that are also input tensors
    std::vector<Tensor*> inputTensors;

    // The subset of tensors that are also output tensors
    std::vector<Tensor*> outputTensors;

    // The definitive list of all nodes in the graph
    std::vector<GraphNode*> nodes;

    // The subset of node that have all of their input tensors ready, but
    // have not yet been evaluated to produce their output tensors.
    // With control flow, a node may appear on this list more than once during its
    // lifetime, although the list itself should only contain unique nodes.
    std::list<GraphNode*> nextNodeList;

    // Maximum number of times to evalute a node before
    // warning.
    const int MAX_EVAL_COUNT = 10000;
};
};    // namespace TosaReference

#endif

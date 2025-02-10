
// Copyright (c) 2020,2024-2025 ARM Limited.
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

#ifndef OPS_CONTROL_FLOW_H
#define OPS_CONTROL_FLOW_H

#include "graph_node.h"

#define MAX_WHILE_LOOP_ITERATION 10000

namespace TosaReference
{
class OpControlFlow : public GraphNode
{
public:
    OpControlFlow(SubgraphTraverser* sgt_, TosaSerializationHandler* tsh_, Op op_, uint64_t id_);
    ~OpControlFlow();

    virtual int evalBlock(TosaSerializationBasicBlock* block,
                          std::vector<TosaReference::Tensor*>& block_inputs,
                          std::vector<TosaReference::Tensor*>& block_outputs);

protected:
    TosaSerializationHandler* tsh;
};

template <int Rank>
class OpCondIf : public OpControlFlow
{
public:
    OpCondIf(SubgraphTraverser* sgt_, TosaSerializationHandler* tsh_, TosaAttributeBase* attribute_, uint64_t id_);
    virtual ~OpCondIf();

    virtual int checkTensorAttributes();
    virtual int eval();

    using TCond = Eigen::Tensor<bool, Rank>;

protected:
    std::unique_ptr<TosaCondIfAttribute> attribute;
    TosaReference::TensorTemplate<TCond>* cond;
    TosaSerializationBasicBlock* then_block;
    TosaSerializationBasicBlock* else_block;
};

class OpWhileLoop : public OpControlFlow
{
public:
    OpWhileLoop(SubgraphTraverser* sgt_, TosaSerializationHandler* tsh_, TosaAttributeBase* attribute_, uint64_t id_);
    virtual ~OpWhileLoop();

    virtual int checkTensorAttributes();
    virtual int eval();

protected:
    std::unique_ptr<TosaWhileLoopAttribute> attribute;
    TosaSerializationBasicBlock* cond_block;
    TosaSerializationBasicBlock* body_block;
};

};    // namespace TosaReference

#endif

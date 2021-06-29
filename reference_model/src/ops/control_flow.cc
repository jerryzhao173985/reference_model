
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

#include "control_flow.h"
#include "subgraph_traverser.h"

using namespace TosaReference;
using namespace Eigen;
using namespace tosa;

OpControlFlow::OpControlFlow(SubgraphTraverser* sgt_, TosaSerializationHandler* tsh_, Op op_, uint64_t id_)
    : GraphNode(sgt_, op_, id_)
{
    tsh = tsh_;
}

OpControlFlow::~OpControlFlow()
{}

int OpControlFlow::evalBlock(TosaSerializationBasicBlock* block,
                             std::vector<TosaReference::Tensor*>& block_inputs,
                             std::vector<TosaReference::Tensor*>& block_outputs)
{
    std::string block_name = block->GetName();

    DEBUG_MED(OP, "Evaluating block %s", block_name.c_str());

    SubgraphTraverser gt(block, tsh);

    if (gt.initializeGraph())
    {
        FATAL_ERROR("Unable to initialize graph traverser for block %s", block_name.c_str());
    }

    if (gt.linkTensorsAndNodes())
    {
        FATAL_ERROR("Failed to link tensors and nodes for block %s", block_name.c_str());
    }

    if (gt.validateGraph())
    {
        FATAL_ERROR("Failed to validate subgraph for block %s", block_name.c_str());
    }

    int num_input_tensors  = gt.getNumInputTensors();
    int num_output_tensors = gt.getNumOutputTensors();

    for (size_t i = 0; i < block_inputs.size(); i++)
    {
        DEBUG_HIGH(OP, "Input[%ld]:  %s", i, block_inputs[i]->getName().c_str());
    }
    for (size_t i = 0; i < block_outputs.size(); i++)
    {
        DEBUG_HIGH(OP, "Output[%ld]: %s", i, block_outputs[i]->getName().c_str());
    }

    ASSERT_MSG((size_t)num_input_tensors == block_inputs.size(),
               "op block %s inputs[%lu] does not match with graph traverser's inputs[%d]", block_name.c_str(),
               block_inputs.size(), num_input_tensors);
    ASSERT_MSG((size_t)num_output_tensors == block_outputs.size(),
               "op block %s outputs[%lu] does not match with graph traverser's outputs[%d]", block_name.c_str(),
               block_outputs.size(), num_output_tensors);

    // set graph traverser's input = basic block's input
    for (int i = 0; i < num_input_tensors; i++)
    {
        TosaReference::Tensor* tensor = gt.getInputTensor(i);
        ASSERT_MSG(!tensor->is_allocated(), "block %s input tensors are unexpectedly initialized before",
                   block_name.c_str());

        if (tensor->allocate())
        {
            WARNING("Fail to allocate tensor %s", tensor->getName().c_str());
            return 1;
        }

        if (tensor->copyValueFrom(block_inputs[i]))
        {
            WARNING("Fail to copy tensor value %s -> %s", block_inputs[i]->getName().c_str(),
                    tensor->getName().c_str());
            return 1;
        }

        tensor->setIsValid();

        // Push ready consumers to the next node list
        for (auto gn : tensor->getConsumers())
        {
            if (gn->hasAllInputsReady() && !gn->getOnNextNodeList())
            {
                gt.addToNextNodeList(gn);
            }
        }
    }

    if (gt.evaluateAll())
    {
        FATAL_ERROR("Error evaluating network.  Giving up.");
    }

    // make sure output tensor is evaluated and show its value
    bool all_output_valid = true;
    for (int i = 0; i < num_output_tensors; i++)
    {
        const TosaReference::Tensor* ct = gt.getOutputTensor(i);
        ASSERT_MEM(ct);
        if (!ct->getIsValid())
        {
            ct->dumpTensorParams(g_func_debug.func_debug_file);
            if (DEBUG_ENABLED(DEBUG_VERB_HIGH, GT))
            {
                ct->dumpTensor(g_func_debug.func_debug_file);
            }
            all_output_valid = false;
        }
    }
    if (!all_output_valid)
    {
        gt.dumpGraph(g_func_debug.func_debug_file);
        FATAL_ERROR("SubgraphTraverser \"%s\" error: Output tensors are not all valid at the end of evaluation.",
                    block_name.c_str());
    }

    // set basic block's output = subgraph_traverser's output
    for (int i = 0; i < num_output_tensors; i++)
    {
        TosaReference::Tensor* tensor = gt.getOutputTensor(i);
        ASSERT_MSG(tensor->is_allocated(), "tensor %s is not allocated", tensor->getName().c_str());

        if (block_outputs[i]->copyValueFrom(tensor))
        {
            WARNING("Fail to copy tensor value %s -> %s", tensor->getName().c_str(), outputs[i]->getName().c_str());
            return 1;
        }
    }
    return 0;
}

OpCondIf::OpCondIf(SubgraphTraverser* sgt_, TosaSerializationHandler* tsh_, TosaAttributeBase* attribute_, uint64_t id_)
    : OpControlFlow(sgt_, tsh_, Op_COND_IF, id_)
{
    INIT_ATTRIBUTE(CondIf);
}

OpCondIf::~OpCondIf()
{
    if (attribute)
        delete attribute;
}

int OpCondIf::checkTensorAttributes()
{
    if (getInputs().size() < 1)
    {
        WARNING("OpCondIf: must have at least 1 operand");
        return 1;
    }

    if (inputs[0]->getDtype() != DType_BOOL || inputs[0]->getRank() != 0)
    {
        WARNING("OpCondIf: invalid tensor dtype=%s, rank=%d", EnumNamesDType()[inputs[0]->getDtype()],
                inputs[0]->getRank());
        return 1;
    }

    cond = dynamic_cast<TosaReference::Tensor0<bool>*>(inputs[0]);
    ASSERT_MEM(cond);

    then_block = tsh->GetBlockByName(attribute->then_branch());
    else_block = tsh->GetBlockByName(attribute->else_branch());

    if (!then_block)
    {
        WARNING("OpCondIf: fail to resolve then_branch %s", attribute->then_branch().c_str());
        return 1;
    }

    if (!else_block)
    {
        WARNING("OpCondIf: fail to resolve else_branch %s", attribute->else_branch().c_str());
        return 1;
    }

    return 0;
}

int OpCondIf::eval()
{
    bool cond_val = cond->getTensor()(0);
    std::vector<TosaReference::Tensor*> block_inputs(getInputs().begin() + 1, getInputs().end());

    if (cond_val)
    {
        if (evalBlock(then_block, block_inputs, getOutputs()))
        {
            WARNING("OpCondIf: Fail to evaluate then branch block %s", attribute->then_branch().c_str());
            return 1;
        }
    }
    else
    {
        if (evalBlock(else_block, block_inputs, getOutputs()))
        {
            WARNING("OpCondIf: Fail to evaluate else branch block %s", attribute->else_branch().c_str());
            return 1;
        }
    }

    return GraphNode::eval();
}

OpWhileLoop::OpWhileLoop(SubgraphTraverser* sgt_,
                         TosaSerializationHandler* tsh_,
                         TosaAttributeBase* attribute_,
                         uint64_t id_)
    : OpControlFlow(sgt_, tsh_, Op_WHILE_LOOP, id_)
{
    INIT_ATTRIBUTE(WhileLoop);
}

OpWhileLoop::~OpWhileLoop()
{
    if (attribute)
        delete attribute;
}

int OpWhileLoop::checkTensorAttributes()
{
    if (getInputs().size() <= 0)
    {
        WARNING("OpWhileLoop: must have at least 1 operands");
        return 1;
    }

    if (getInputs().size() != getOutputs().size())
    {
        WARNING("OpWhileLoop: inputs and outputs size must match");
        return 1;
    }

    cond_block = tsh->GetBlockByName(attribute->cond_branch());
    body_block = tsh->GetBlockByName(attribute->body_branch());

    if (!cond_block)
    {
        WARNING("OpWhileLoop: fail to resolve cond_branch %s", attribute->cond_branch().c_str());
        return 1;
    }

    if (!body_block)
    {
        WARNING("OpWhileLoop: fail to resolve body_branch %s", attribute->body_branch().c_str());
        return 1;
    }

    if (cond_block->GetOutputs().size() != 1)
    {
        WARNING("OpWhileLoop: invalid cond_block output size %lu", cond_block->GetOutputs().size());
        return 1;
    }

    TosaSerializationTensor* cond_output_tensor = cond_block->GetTensorByName(cond_block->GetOutputs()[0]);

    if (!cond_output_tensor)
    {
        WARNING("OpWhileLoop: fail to resolve cond_block's output tensor %s", cond_block->GetOutputs()[0].c_str());
        return 1;
    }

    if (cond_output_tensor->GetDtype() != DType_BOOL)
    {
        WARNING("OpWhileLoop: invalid cond_block's output tensor data type %s",
                EnumNamesDType()[cond_output_tensor->GetDtype()]);
        return 1;
    }
    if (cond_output_tensor->GetShape().size() != 0)
    {
        WARNING("OpWhileLoop: invalid cond_block's output rank %lu", cond_output_tensor->GetShape().size());
        return 1;
    }

    return 0;
}

int OpWhileLoop::eval()
{

    TosaReference::Tensor0<bool> cond_output_ctensor(std::string("cond_output"), DType_BOOL, std::vector<int32_t>({}));

    cond_output_ctensor.allocate();
    std::vector<TosaReference::Tensor*> cond_block_outputs;
    cond_block_outputs.push_back(&cond_output_ctensor);

    size_t num_input_output = getInputs().size();
    size_t eval_count       = 0;

    while (eval_count++ < MAX_WHILE_LOOP_ITERATION)
    {
        if (evalBlock(cond_block, getInputs(), cond_block_outputs))
        {
            WARNING("OpWhileLoop: Fail to evaluate cond block %s", attribute->cond_branch().c_str());
            return 1;
        }
        bool cond_val = cond_output_ctensor.getTensor()(0);
        DEBUG_HIGH(OP, "Conditional block value: %d", cond_val);

        if (cond_val)
        {
            if (evalBlock(body_block, getInputs(), getOutputs()))
            {
                WARNING("OpWhileLoop: Fail to evaluate body block %s", attribute->body_branch().c_str());
                return 1;
            }

            // assigning output tensors value back to input tensors value for next iteration
            for (size_t i = 0; i < num_input_output; i++)
            {
                if (getInputs()[i]->copyValueFrom(getOutputs()[i]))
                {
                    WARNING("Fail to copy tensor value %s -> %s", getOutputs()[i]->getName().c_str(),
                            getInputs()[i]->getName().c_str());
                    return 1;
                }
            }
        }
        else
        {
            // in last iteration or the case it never evaluates body block
            // assign input tensors value to output tensors
            for (size_t i = 0; i < num_input_output; i++)
            {
                if (getOutputs()[i]->copyValueFrom(getInputs()[i]))
                {
                    WARNING("Fail to copy tensor value %s -> %s", getInputs()[i]->getName().c_str(),
                            getOutputs()[i]->getName().c_str());
                    return 1;
                }
            }
            break;
        }
    }

    return GraphNode::eval();
}

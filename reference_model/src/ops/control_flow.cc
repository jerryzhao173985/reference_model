
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

    SubgraphTraverser block_sgt(block, tsh, this->parent_sgt);

    ERROR_IF(block_sgt.initializeGraph(), "evalBlock(): Unable to initialize graph traverser for %s",
             block_name.c_str());
    ERROR_IF(block_sgt.linkTensorsAndNodes(), "evalBlock(): Failed to link tensors and nodes for %s",
             block_name.c_str());
    ERROR_IF(block_sgt.validateGraph(), "evalBlock(): Failed to validate subgraph for %s", block_name.c_str());
    ERROR_IF(block_sgt.allocateTensor(), "evalBlock(): Failed to allocate tensor for %s", block_name.c_str());

    int num_input_tensors  = block_sgt.getNumInputTensors();
    int num_output_tensors = block_sgt.getNumOutputTensors();

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
        TosaReference::Tensor* tensor = block_sgt.getInputTensor(i);
        ERROR_IF(!tensor->is_allocated(), "block %s input tensor %s are not initialized before use", block_name.c_str(),
                 tensor->getName().c_str());

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
                block_sgt.addToNextNodeList(gn);
            }
        }
    }

    ERROR_IF(block_sgt.evaluateAll(), "Error evaluating network.  Giving up.");

    // pass block status back
    switch (block_sgt.getGraphStatus())
    {
        case GraphStatus::TOSA_VALID:
        {
            DEBUG_MED(OP, "Successfully evaluating block %s", block_name.c_str());
            break;
        }
        case GraphStatus::TOSA_UNPREDICTABLE:
        {
            DEBUG_MED(OP, "Finish evaluating block %s but result is UNPREDICTABLE", block_name.c_str());
            DEBUG_MED(OP, "Setting parent graph status to UNPREDICTABLE");
            parent_sgt->setGraphStatus(GraphStatus::TOSA_UNPREDICTABLE);
            break;
        }
        case GraphStatus::TOSA_ERROR:
        {
            DEBUG_MED(OP, "Fail evaluating block %s. Result is ERROR", block_name.c_str());
            if (parent_sgt->getGraphStatus() != GraphStatus::TOSA_UNPREDICTABLE)
            {
                DEBUG_MED(OP, "Setting parent graph status to ERROR");
                parent_sgt->setGraphStatus(GraphStatus::TOSA_ERROR);
                return 1;
            }
        }
    }

    // make sure output tensor is evaluated and show its value
    bool all_output_valid = true;
    for (int i = 0; i < num_output_tensors; i++)
    {
        const TosaReference::Tensor* ct = block_sgt.getOutputTensor(i);
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
        block_sgt.dumpGraph(g_func_debug.func_debug_file);
        ERROR_IF(true, "SubgraphTraverser \"%s\" error: Output tensors are not all valid at the end of evaluation.",
                 block_name.c_str());
    }

    // set basic block's output = subgraph_traverser's output
    for (int i = 0; i < num_output_tensors; i++)
    {
        TosaReference::Tensor* tensor = block_sgt.getOutputTensor(i);
        ERROR_IF(!tensor->is_allocated(), "block %s input tensor %s are not initialized before use", block_name.c_str(),
                 tensor->getName().c_str());

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
    ERROR_IF(!tsh, "OpCondIf: tosa serialization handler must not be null");

    ERROR_IF(getInputs().size() < 1, "OpCondIf: must have at least 1 operand");

    ERROR_IF(inputs[0]->getDtype() != TOSA_REF_TYPE_BOOL || inputs[0]->getRank() != 0,
             "OpCondIf: invalid tensor dtype=%s, rank=%d", EnumNameTOSAREFTYPE(inputs[0]->getDtype()),
             inputs[0]->getRank());

    cond = dynamic_cast<TosaReference::Tensor0<bool>*>(inputs[0]);
    ASSERT_MEM(cond);

    auto then_region = tsh->GetRegionByName(attribute->then_branch());
    auto else_region = tsh->GetRegionByName(attribute->else_branch());
    if (then_region && else_region)
    {
        // new serialization: then_branch and else_branch point to regions
        then_block = then_region->GetBlocks().front();
        else_block = else_region->GetBlocks().front();
    }
    else
    {
        // old serialization: then_branch and else_branch point to blocks in curr_region
        auto region_name = getParentSGT()->getRegionName();
        auto curr_region = tsh->GetRegionByName(region_name);
        then_block       = curr_region->GetBlockByName(attribute->then_branch());
        else_block       = curr_region->GetBlockByName(attribute->else_branch());
    }

    ERROR_IF(!then_block, "OpCondIf: fail to resolve then_branch %s", attribute->then_branch().c_str());

    ERROR_IF(!else_block, "OpCondIf: fail to resolve else_branch %s", attribute->else_branch().c_str());

    // Make sure operator input/output matches block input/output
    // Skip the first rank 0 bool tensor on input list
    int32_t num_input_tensor  = getInputs().size() - 1;
    int32_t num_output_tensor = getOutputs().size();

    ERROR_IF((int32_t)then_block->GetInputs().size() != num_input_tensor,
             "OpCondIf: then_block has unexpected number of input");
    ERROR_IF((int32_t)else_block->GetInputs().size() != num_input_tensor,
             "OpCondIf: else_block has unexpected number of input");
    ERROR_IF((int32_t)then_block->GetOutputs().size() != num_output_tensor,
             "OpCondIf: then_block has unexpected number of output");
    ERROR_IF((int32_t)else_block->GetOutputs().size() != num_output_tensor,
             "OpCondIf: else_block has unexpected number of output");

    for (int32_t i = 0; i < num_input_tensor; i++)
    {
        Tensor* operator_input                    = getInputs()[i + 1];
        std::string then_block_input_name         = then_block->GetInputs()[i];
        std::string else_block_input_name         = else_block->GetInputs()[i];
        TosaSerializationTensor* then_block_input = then_block->GetTensorByName(then_block_input_name);
        TosaSerializationTensor* else_block_input = else_block->GetTensorByName(else_block_input_name);
        ERROR_IF(operator_input->getDtype() != ConvertDType(then_block_input->GetDtype()),
                 "OpCondIf: input tensor type mismatch with then_block input type");
        ERROR_IF(operator_input->getDtype() != ConvertDType(else_block_input->GetDtype()),
                 "OpCondIf: input tensor type mismatch with else_block input type");
        ERROR_IF(operator_input->getRank() != (int32_t)then_block_input->GetShape().size(),
                 "OpCondIf: input tensor rank mismatch with then_block input rank");
        ERROR_IF(operator_input->getRank() != (int32_t)else_block_input->GetShape().size(),
                 "OpCondIf: input tensor rank mismatch with else_block input rank");
        for (int32_t d = 0; d < operator_input->getRank(); d++)
        {
            ERROR_IF(operator_input->getShape()[d] != then_block_input->GetShape()[d],
                     "OpCondIf: input tensor dimension mismatch with then_block input dimension");
            ERROR_IF(operator_input->getShape()[d] != else_block_input->GetShape()[d],
                     "OpCondIf: input tensor dimension mismatch with else_block input dimension");
        }
    }

    for (int32_t i = 0; i < num_output_tensor; i++)
    {
        Tensor* operator_output                    = getOutputs()[i];
        std::string then_block_output_name         = then_block->GetOutputs()[i];
        std::string else_block_output_name         = else_block->GetOutputs()[i];
        TosaSerializationTensor* then_block_output = then_block->GetTensorByName(then_block_output_name);
        TosaSerializationTensor* else_block_output = else_block->GetTensorByName(else_block_output_name);
        ERROR_IF(operator_output->getDtype() != ConvertDType(then_block_output->GetDtype()),
                 "OpCondIf: output tensor type mismatch with then_block output type");
        ERROR_IF(operator_output->getDtype() != ConvertDType(else_block_output->GetDtype()),
                 "OpCondIf: output tensor type mismatch with else_block output type");
        ERROR_IF(operator_output->getRank() != (int32_t)then_block_output->GetShape().size(),
                 "OpCondIf: output tensor rank mismatch with then_block output rank");
        ERROR_IF(operator_output->getRank() != (int32_t)else_block_output->GetShape().size(),
                 "OpCondIf: output tensor rank mismatch with else_block output rank");
        for (int32_t d = 0; d < operator_output->getRank(); d++)
        {
            ERROR_IF(operator_output->getShape()[d] != then_block_output->GetShape()[d],
                     "OpCondIf: output tensor dimension mismatch with then_block output dimension");
            ERROR_IF(operator_output->getShape()[d] != else_block_output->GetShape()[d],
                     "OpCondIf: output tensor dimension mismatch with else_block output dimension");
        }
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
    if (!tsh) {
        WARNING("OpWhileLoop: tosa serialization handler must not be null");
        return 1;
    }

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

    auto cond_region = tsh->GetRegionByName(attribute->cond_branch());
    auto body_region = tsh->GetRegionByName(attribute->body_branch());
    if (cond_region && body_region)
    {
        // new serialization: then_branch and else_branch point to regions
        cond_block = cond_region->GetBlocks().front();
        body_block = body_region->GetBlocks().front();
    }
    else
    {
        auto region_name = getParentSGT()->getRegionName();
        auto curr_region = tsh->GetRegionByName(region_name);
        cond_block       = curr_region->GetBlockByName(attribute->cond_branch());
        body_block       = curr_region->GetBlockByName(attribute->body_branch());
    }

    ERROR_IF(!cond_block, "OpWhileLoop: fail to resolve cond_branch %s", attribute->cond_branch().c_str());
    ERROR_IF(!body_block, "OpWhileLoop: fail to resolve body_branch %s", attribute->body_branch().c_str());

    // Make sure operator input/output matches block input/output
    int32_t num_block_tensor = getInputs().size();
    ERROR_IF((int32_t)getOutputs().size() != num_block_tensor,
             "OpWhileLoop: operator input tensor doesn't match output");
    ERROR_IF((int32_t)cond_block->GetInputs().size() != num_block_tensor,
             "OpWhileLoop: cond_block has unexpected number of input");
    ERROR_IF((int32_t)body_block->GetInputs().size() != num_block_tensor,
             "OpWhileLoop: body_block has unexpected number of input");
    ERROR_IF((int32_t)body_block->GetOutputs().size() != num_block_tensor,
             "OpWhileLoop: body_block has unexpected number of output");
    for (int32_t i = 0; i < num_block_tensor; i++)
    {
        Tensor* operator_input  = getInputs()[i];
        Tensor* operator_output = getOutputs()[i];
        ERROR_IF(operator_input->matchRankTypeShape(*operator_output),
                 "OpWhileLoop: operator input tensor mismatch operator output tensor");

        std::string cond_block_input_name          = cond_block->GetInputs()[i];
        std::string body_block_input_name          = body_block->GetInputs()[i];
        std::string body_block_output_name         = body_block->GetOutputs()[i];
        TosaSerializationTensor* cond_block_input  = cond_block->GetTensorByName(cond_block_input_name);
        TosaSerializationTensor* body_block_input  = body_block->GetTensorByName(body_block_input_name);
        TosaSerializationTensor* body_block_output = body_block->GetTensorByName(body_block_output_name);

        ERROR_IF(operator_input->getDtype() != ConvertDType(cond_block_input->GetDtype()),
                 "OpWhileLoop: input tensor type mismatch with cond_block input type");
        ERROR_IF(operator_input->getDtype() != ConvertDType(body_block_input->GetDtype()),
                 "OpWhileLoop: input tensor type mismatch with body_block input type");
        ERROR_IF(operator_input->getDtype() != ConvertDType(body_block_output->GetDtype()),
                 "OpWhileLoop: input tensor type mismatch with body_block output type");
        ERROR_IF(operator_input->getRank() != (int32_t)cond_block_input->GetShape().size(),
                 "OpWhileLoop: input tensor rank mismatch with cond_block input rank");
        ERROR_IF(operator_input->getRank() != (int32_t)body_block_input->GetShape().size(),
                 "OpWhileLoop: input tensor rank mismatch with body_block input rank");
        ERROR_IF(operator_input->getRank() != (int32_t)body_block_output->GetShape().size(),
                 "OpWhileLoop: input tensor rank mismatch with body_block output rank");

        for (int32_t d = 0; d < operator_input->getRank(); d++)
        {
            ERROR_IF(operator_input->getShape()[d] != cond_block_input->GetShape()[d],
                     "OpWhileLoop: input tensor dimension mismatch with cond_block input dimension");
            ERROR_IF(operator_input->getShape()[d] != body_block_input->GetShape()[d],
                     "OpWhileLoop: input tensor dimension mismatch with body_block input dimension");
            ERROR_IF(operator_input->getShape()[d] != body_block_output->GetShape()[d],
                     "OpWhileLoop: input tensor dimension mismatch with body_block output dimension");
        }
    }

    ERROR_IF(cond_block->GetOutputs().size() != 1, "OpWhileLoop: cond_block can only have 1 output tensor");
    std::string cond_block_output_name         = cond_block->GetOutputs()[0];
    TosaSerializationTensor* cond_block_output = cond_block->GetTensorByName(cond_block_output_name);
    ERROR_IF(cond_block_output->GetDtype() != DType_BOOL, "OpWhileLoop: cond_block output can only be bool type");
    ERROR_IF(cond_block_output->GetShape().size() != 0, "OpWhileLoop: cond_block output can only be rank 0");

    return 0;
}

int OpWhileLoop::eval()
{
    TosaReference::Tensor0<bool> cond_output_ctensor("cond_output", DType_BOOL, std::vector<int32_t>({}));

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
                getInputs()[i] = getOutputs()[i];
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

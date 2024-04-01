// Copyright (c) 2023-2024, ARM Limited.
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

#include "shape.h"

using namespace TosaReference;
using namespace Eigen;
using namespace tosa;

OpConstShape::OpConstShape(SubgraphTraverser* sgt_, uint64_t id_)
    : GraphNode(sgt_, Op_CONST, id_)
{
    setRequiredOperands(0, 1);
}

OpConstShape::~OpConstShape()
{}

int OpConstShape::checkTensorAttributes()
{
    if (validateRequiredOperands())
        return 1;

    return 0;
}

int OpConstShape::eval()
{
    // set the shapeValue given the actual tensor value
    using EigenType = typename GetEigenType<TOSA_REF_TYPE_SHAPE>::type;
    auto out        = dynamic_cast<TosaReference::TensorTemplate<Eigen::Tensor<EigenType, 1>>*>(this->getOutputs()[0]);

    std::vector<int> shapeValue;
    for (int i = 0; out != nullptr && i < out->getTensor().size(); ++i)
    {
        shapeValue.push_back(out->getTensor()(i));
    }

    this->getOutputs()[0]->setShapeValue(shapeValue);

    for (auto ct : getOutputs())
    {
        if (!ct->getIsValid())
        {
            std::string err = "Constant Shape tensor " + ct->getName() + " not correctly initialized";
            printNodeValidationError(err.c_str());
            return 1;
        }
    }

    // Evaluation is trivial for constants
    return GraphNode::eval();
}

OpConcatShape::OpConcatShape(SubgraphTraverser* sgt_, uint64_t id_)
    : GraphNode(sgt_, Op_CONCAT_SHAPE, id_)
{
    setRequiredOperands(-1, 1);
    setRequiredRank(1, 1);
}

OpConcatShape::~OpConcatShape()
{}

int OpConcatShape::checkTensorAttributes()
{
    if (validateRequiredOperands())
        return 1;

    if (inputs.empty())
    {
        printNodeValidationError("ConcatShape operator must have at least one input tensor");
        return 1;
    }

    int32_t num_inputs     = inputs.size();
    int32_t elements_count = 0;
    for (int32_t i = 0; i < num_inputs; i++)
    {
        if (validateRequiredRank(inputs[i]))
            return 1;
        ins.push_back(dynamic_cast<TosaReference::TensorTemplate<TIn>*>(inputs[i]));
        elements_count += inputs[i]->getShape()[0];
    }

    ERROR_IF(elements_count != outputs[0]->getShape()[0],
             "OpConcatShape: sum of input elements not equal to output number of elements");

    num_dims = outputs[0]->getShape()[0];
    out      = dynamic_cast<TosaReference::TensorTemplate<TOut>*>(outputs[0]);

    return 0;
}

int OpConcatShape::eval()
{
    ETensor1<EigenType> out_tensor(num_dims);
    int32_t out_idx = 0;
    for (size_t i = 0; i < ins.size(); i++)
    {
        // all tosa.shape values are 1-d tensors
        // interate in_idx in range of [0, rank of 1-d tensor]
        for (int32_t in_idx = 0; in_idx < inputs[i]->getShape()[0]; in_idx++)
        {
            out_tensor(out_idx) = ins[i]->getTensor()(in_idx);
            out_idx++;
        }
    }
    out->getTensor() = out_tensor;

    // set the shapeValue given the actual tensor value
    std::vector<int> shapeValue;
    for (int i = 0; i < out->getTensor().size(); ++i)
    {
        shapeValue.push_back(out->getTensor()(i));
    }
    this->getOutputs()[0]->setShapeValue(shapeValue);

    return GraphNode::eval();
}

ShapeBinaryNodeBase::ShapeBinaryNodeBase(SubgraphTraverser* sgt_, const Op& op_, uint64_t id_)
    : GraphNode(sgt_, op_, id_)
{
    setRequiredOperands(2, 1);
    setRequiredRank(1, 1);

    fcn = [](EigenType a, EigenType b) -> EigenType { return EigenType(); };
}

ShapeBinaryNodeBase::~ShapeBinaryNodeBase()
{}

int ShapeBinaryNodeBase::checkTensorAttributes()
{
    if (validateRequiredOperands())
        return 1;
    if (validateRequiredRank(inputs[0]) || validateRequiredRank(inputs[1]) || validateRequiredRank(outputs[0]))
        return 1;

    num_dims = outputs[0]->getShape()[0];

    if (inputs[0]->getShape()[0] != num_dims)
    {
        std::string err = "Binary shape operators " + std::string(EnumNamesOp()[nodeType]) +
                          " lhs input and output rank/shape must match";
        printNodeValidationError(err.c_str());
        return 1;
    }

    if (inputs[1]->getShape()[0] != num_dims)
    {
        std::string err = "Binary shape operators " + std::string(EnumNamesOp()[nodeType]) +
                          " rhs input and output rank/shape must match";
        printNodeValidationError(err.c_str());
        return 1;
    }

    a      = dynamic_cast<TosaReference::TensorTemplate<TIn>*>(inputs[0]);
    b      = dynamic_cast<TosaReference::TensorTemplate<TIn>*>(inputs[1]);
    result = dynamic_cast<TosaReference::TensorTemplate<TOut>*>(outputs[0]);

    ASSERT_MEM(a && b && result);

    return 0;
}

int ShapeBinaryNodeBase::eval()
{
    auto ia = a->getTensor();
    auto ib = b->getTensor();
    ETensor1<EigenType> out_tens(num_dims);
    for (int32_t i = 0; i < num_dims; i++)
    {
        EigenType lhs = ia(i);
        EigenType rhs = ib(i);
        out_tens(i)   = (lhs < 0 || rhs < 0) ? static_cast<EigenType>(-1) : fcn(lhs, rhs);
    }

    result->getTensor() = out_tens;

    // set the shapeValue given the actual tensor value
    std::vector<int> shapeValue;
    for (int i = 0; i < result->getTensor().size(); ++i)
    {
        shapeValue.push_back(result->getTensor()(i));
    }
    this->getOutputs()[0]->setShapeValue(shapeValue);

    return GraphNode::eval();
}

int OpAddShape::register_fcn()
{
    fcn = [](EigenType a, EigenType b) -> EigenType { return a + b; };
    return 0;
}

int OpSubShape::register_fcn()
{
    fcn = [](EigenType a, EigenType b) -> EigenType { return a - b; };
    return 0;
}

int OpMulShape::register_fcn()
{
    fcn = [](EigenType a, EigenType b) -> EigenType { return a * b; };
    return 0;
}

int OpDivShape::register_fcn()
{
    fcn = [](EigenType a, EigenType b) -> EigenType {
        return (b == static_cast<EigenType>(0)) ? static_cast<EigenType>(-1) : (a / b);
    };
    return 0;
}
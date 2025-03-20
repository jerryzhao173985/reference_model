// Copyright (c) 2023-2025, ARM Limited.
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
        shapeValue.push_back(static_cast<int>(out->getTensor()(i)));
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

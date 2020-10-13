
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

#include "custom.h"

using namespace TosaReference;
using namespace Eigen;
using namespace tosa;

OpCustom::OpCustom(uint64_t id_)
    : GraphNode(Op_CUSTOM, id_)
{}

OpCustom::~OpCustom()
{}

int OpCustom::checkTensorAttributes()
{
    return 0;
}

int OpCustom::eval()
{
    FATAL_ERROR_NODE("not supported yet");

    // Evaluation is trivial for constants
    return GraphNode::eval();
}

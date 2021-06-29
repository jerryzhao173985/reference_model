
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

#ifndef OPS_CUSTOM_H
#define OPS_CUSTOM_H

#include "graph_node.h"

using namespace tosa;

namespace TosaReference
{

class OpCustom : public GraphNode
{
public:
    OpCustom(SubgraphTraverser* sgt_, uint64_t id_);
    virtual ~OpCustom();

    virtual int checkTensorAttributes();
    virtual int eval();
};

};    // namespace TosaReference

#endif

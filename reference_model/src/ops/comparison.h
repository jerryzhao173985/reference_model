
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

#ifndef OPS_COMPARISON_H
#define OPS_COMPARISON_H

#include "ewise_binary.h"
#include "graph_node.h"

using namespace tosa;

namespace TosaReference
{

template <int Rank, TOSA_REF_TYPE Dtype>
class OpEqual : public BinaryNode<Rank, Dtype, TOSA_REF_TYPE_BOOL>
{
public:
    OpEqual(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_)
        : BinaryNode<Rank, Dtype, TOSA_REF_TYPE_BOOL>(sgt_, Op_EQUAL, id_)
    {
        register_fcn();
    }
    using InEigenType  = typename GetEigenType<Dtype>::type;
    using OutEigenType = typename GetEigenType<TOSA_REF_TYPE_BOOL>::type;
    virtual int register_fcn();
};

template <int Rank, TOSA_REF_TYPE Dtype>
class OpGreater : public BinaryNode<Rank, Dtype, TOSA_REF_TYPE_BOOL>
{
public:
    OpGreater(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_)
        : BinaryNode<Rank, Dtype, TOSA_REF_TYPE_BOOL>(sgt_, Op_GREATER, id_)
    {
        register_fcn();
    }
    using InEigenType  = typename GetEigenType<Dtype>::type;
    using OutEigenType = typename GetEigenType<TOSA_REF_TYPE_BOOL>::type;
    virtual int register_fcn();
};

template <int Rank, TOSA_REF_TYPE Dtype>
class OpGreaterEqual : public BinaryNode<Rank, Dtype, TOSA_REF_TYPE_BOOL>
{
public:
    OpGreaterEqual(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_)
        : BinaryNode<Rank, Dtype, TOSA_REF_TYPE_BOOL>(sgt_, Op_EQUAL, id_)
    {
        register_fcn();
    }
    using InEigenType  = typename GetEigenType<Dtype>::type;
    using OutEigenType = typename GetEigenType<TOSA_REF_TYPE_BOOL>::type;
    virtual int register_fcn();
};

};    // namespace TosaReference

#endif

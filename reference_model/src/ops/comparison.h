
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

#ifndef OPS_COMPARISON_H
#define OPS_COMPARISON_H

#include "ewise_binary.h"
#include "graph_node.h"

using namespace tosa;

namespace TosaReference
{

template <int Rank, DType Dtype>
class OpEqual : public BinaryNode<Rank, Dtype, DType_BOOL>
{
public:
    OpEqual(TosaAttributeBase* attribute_, TosaQuantInfoBase* qinfo_, uint64_t id_)
        : BinaryNode<Rank, Dtype, DType_BOOL>(Op_EQUAL, qinfo_, id_)
    {
        register_fcn();
    }
    using InEigenType  = typename GetEigenType<Dtype>::type;
    using OutEigenType = typename GetEigenType<DType_BOOL>::type;
    virtual int register_fcn();
};

template <int Rank, DType Dtype>
class OpGreater : public BinaryNode<Rank, Dtype, DType_BOOL>
{
public:
    OpGreater(TosaAttributeBase* attribute_, TosaQuantInfoBase* qinfo_, uint64_t id_)
        : BinaryNode<Rank, Dtype, DType_BOOL>(Op_GREATER, qinfo_, id_)
    {
        register_fcn();
    }
    using InEigenType  = typename GetEigenType<Dtype>::type;
    using OutEigenType = typename GetEigenType<DType_BOOL>::type;
    virtual int register_fcn();
};

template <int Rank, DType Dtype>
class OpGreaterEqual : public BinaryNode<Rank, Dtype, DType_BOOL>
{
public:
    OpGreaterEqual(TosaAttributeBase* attribute_, TosaQuantInfoBase* qinfo_, uint64_t id_)
        : BinaryNode<Rank, Dtype, DType_BOOL>(Op_EQUAL, qinfo_, id_)
    {
        register_fcn();
    }
    using InEigenType  = typename GetEigenType<Dtype>::type;
    using OutEigenType = typename GetEigenType<DType_BOOL>::type;
    virtual int register_fcn();
};

};    // namespace TosaReference

#endif

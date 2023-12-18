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

#ifndef OPS_SHAPES_H
#define OPS_SHAPES_H

#include "graph_node.h"

namespace TosaReference
{

class OpConstShape : public GraphNode
{
public:
    OpConstShape(SubgraphTraverser* sgt_, uint64_t id_);
    virtual ~OpConstShape();

    virtual int checkTensorAttributes();
    virtual int eval();
};

class OpConcatShape : public GraphNode
{
public:
    OpConcatShape(SubgraphTraverser* sgt_, uint64_t id_);
    virtual ~OpConcatShape();

    virtual int checkTensorAttributes();
    virtual int eval();

    using EigenType = typename GetEigenType<TOSA_REF_TYPE_SHAPE>::type;
    using TIn       = Eigen::Tensor<EigenType, 1>;
    using TOut      = Eigen::Tensor<EigenType, 1>;

protected:
    int32_t num_dims;    // number of dimensions in concat_shape output
    std::vector<TosaReference::TensorTemplate<TIn>*> ins;
    TosaReference::TensorTemplate<TOut>* out;
};

class ShapeBinaryNodeBase : public GraphNode
{
public:
    ShapeBinaryNodeBase(SubgraphTraverser* sgt_, const Op& op_, uint64_t id_);
    virtual ~ShapeBinaryNodeBase();

    virtual int checkTensorAttributes() final;
    virtual int eval();
    virtual int register_fcn() = 0;

    using EigenType = typename GetEigenType<TOSA_REF_TYPE_SHAPE>::type;
    using TIn       = Eigen::Tensor<EigenType, 1>;
    using TOut      = Eigen::Tensor<EigenType, 1>;

protected:
    int32_t num_dims;    // number of dimensions in shape op's result
    std::function<EigenType(EigenType, EigenType)> fcn;
    TosaReference::TensorTemplate<TIn>* a;
    TosaReference::TensorTemplate<TIn>* b;
    TosaReference::TensorTemplate<TOut>* result;
};

class OpAddShape : public ShapeBinaryNodeBase
{
public:
    OpAddShape(SubgraphTraverser* sgt_, uint64_t id_)
        : ShapeBinaryNodeBase(sgt_, Op_ADD_SHAPE, id_)
    {
        register_fcn();
    }
    virtual int register_fcn();
};

class OpSubShape : public ShapeBinaryNodeBase
{
public:
    OpSubShape(SubgraphTraverser* sgt_, uint64_t id_)
        : ShapeBinaryNodeBase(sgt_, Op_SUB_SHAPE, id_)
    {
        register_fcn();
    }
    virtual int register_fcn();
};

class OpMulShape : public ShapeBinaryNodeBase
{
public:
    OpMulShape(SubgraphTraverser* sgt_, uint64_t id_)
        : ShapeBinaryNodeBase(sgt_, Op_MUL_SHAPE, id_)
    {
        register_fcn();
    }
    virtual int register_fcn();
};

class OpDivShape : public ShapeBinaryNodeBase
{
public:
    OpDivShape(SubgraphTraverser* sgt_, uint64_t id_)
        : ShapeBinaryNodeBase(sgt_, Op_DIV_SHAPE, id_)
    {
        register_fcn();
    }
    virtual int register_fcn();
};

};    // namespace TosaReference

#endif

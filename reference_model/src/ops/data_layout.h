
// Copyright (c) 2020-2025, ARM Limited.
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

#ifndef OPS_DATA_LAYOUT_H
#define OPS_DATA_LAYOUT_H

#include "graph_node.h"

using namespace tosa;

namespace TosaReference
{

template <int Rank, TOSA_REF_TYPE Dtype>
class OpConcat : public GraphNode
{
public:
    OpConcat(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_);
    virtual ~OpConcat();

    virtual int checkTensorAttributes();
    virtual int eval();

    using InEigenType  = typename GetEigenType<Dtype>::type;
    using OutEigenType = typename GetEigenType<Dtype>::type;
    using TIn          = Eigen::Tensor<InEigenType, Rank>;
    using TOut         = Eigen::Tensor<OutEigenType, Rank>;

protected:
    Eigen::array<int, Rank> reverser;
    std::vector<TosaReference::TensorTemplate<TIn>*> ins;
    std::unique_ptr<TosaConcatAttribute> attribute;
    TosaReference::TensorTemplate<TOut>* out;
};

template <int Rank, TOSA_REF_TYPE Dtype>
class OpPad : public GraphNode
{
public:
    OpPad(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_);
    virtual ~OpPad();
    virtual int checkTensorAttributes();
    virtual int eval();

    using InEigenType      = typename GetEigenType<Dtype>::type;
    using InEigenShapeType = typename GetEigenType<TOSA_REF_TYPE_SHAPE>::type;
    using OutEigenType     = typename GetEigenType<Dtype>::type;
    using TIn              = Eigen::Tensor<InEigenType, Rank>;
    using TPadding         = Eigen::Tensor<InEigenShapeType, 1>;
    using TOut             = Eigen::Tensor<OutEigenType, Rank>;

protected:
    Eigen::array<std::pair<ptrdiff_t, ptrdiff_t>, Rank> paddings_array;
    TosaReference::TensorTemplate<TIn>* in;
    TosaReference::TensorTemplate<TPadding>* padding;
    TosaReference::TensorTemplate<TOut>* out;
    std::unique_ptr<TosaPadAttribute> attribute;
};

template <int InRank, int OutRank, TOSA_REF_TYPE Dtype>
class OpReshape : public GraphNode
{
public:
    OpReshape(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_);
    virtual ~OpReshape();

    virtual int checkTensorAttributes();
    virtual int eval();

    using InEigenType  = typename GetEigenType<Dtype>::type;
    using OutEigenType = typename GetEigenType<Dtype>::type;
    using TIn          = Eigen::Tensor<InEigenType, InRank>;
    using TOut         = Eigen::Tensor<OutEigenType, OutRank>;

protected:
    Eigen::array<Eigen::Index, OutRank> array_shape;
    Eigen::array<Eigen::Index, InRank> in_reverser;
    Eigen::array<Eigen::Index, OutRank> out_reverser;
    TosaReference::TensorTemplate<TIn>* in;
    TosaReference::TensorTemplate<TOut>* out;
};

template <int Rank, TOSA_REF_TYPE Dtype>
class OpReverse : public GraphNode
{
public:
    OpReverse(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_);
    virtual ~OpReverse();

    virtual int checkTensorAttributes();
    virtual int eval();

    using InEigenType  = typename GetEigenType<Dtype>::type;
    using OutEigenType = typename GetEigenType<Dtype>::type;
    using TIn          = Eigen::Tensor<InEigenType, Rank>;
    using TOut         = Eigen::Tensor<OutEigenType, Rank>;

protected:
    std::unique_ptr<TosaReverseAttribute> attribute;
    TosaReference::TensorTemplate<TIn>* in;
    TosaReference::TensorTemplate<TOut>* out;
    Eigen::array<bool, Rank> reverse_array;
};

template <int Rank, TOSA_REF_TYPE Dtype>
class OpSlice : public GraphNode
{
public:
    OpSlice(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_);
    virtual ~OpSlice();

    virtual int checkTensorAttributes();
    virtual int eval();

    using InEigenType      = typename GetEigenType<Dtype>::type;
    using InEigenShapeType = typename GetEigenType<TOSA_REF_TYPE_SHAPE>::type;
    using OutEigenType     = typename GetEigenType<Dtype>::type;
    using TIn              = Eigen::Tensor<InEigenType, Rank>;
    using TSlicing         = Eigen::Tensor<InEigenShapeType, 1>;
    using TOut             = Eigen::Tensor<OutEigenType, Rank>;

protected:
    Eigen::array<Eigen::Index, Rank> begin_array;
    Eigen::array<Eigen::Index, Rank> size_array;
    TosaReference::TensorTemplate<TSlicing>* start;
    TosaReference::TensorTemplate<TSlicing>* size;
    TosaReference::TensorTemplate<TIn>* in;
    TosaReference::TensorTemplate<TOut>* out;
};

template <int Rank, TOSA_REF_TYPE Dtype>
class OpTileBase : public GraphNode
{
public:
    OpTileBase(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_);
    virtual ~OpTileBase();

    virtual int checkTensorAttributes();

    using InEigenType      = typename GetEigenType<Dtype>::type;
    using InEigenShapeType = typename GetEigenType<TOSA_REF_TYPE_SHAPE>::type;
    using OutEigenType     = typename GetEigenType<Dtype>::type;
    using TIn              = Eigen::Tensor<InEigenType, Rank>;
    using TInMultiples     = Eigen::Tensor<InEigenShapeType, 1>;
    using TOut             = Eigen::Tensor<OutEigenType, Rank>;

protected:
    TosaReference::TensorTemplate<TIn>* in;
    TosaReference::TensorTemplate<TInMultiples>* multiples;
    TosaReference::TensorTemplate<TOut>* out;
};

// primary template for op tile
template <int Rank, TOSA_REF_TYPE Dtype>
class OpTile : public OpTileBase<Rank, Dtype>
{
public:
    OpTile(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_)
        : OpTileBase<Rank, Dtype>(sgt_, attribute_, id_)
    {}

protected:
    virtual int eval();
};

// partial specialization for specific rank
#define DEF_OP_TILE_RANK(N)                                                                                            \
    template <TOSA_REF_TYPE Dtype>                                                                                     \
    class OpTile<N, Dtype> : public OpTileBase<N, Dtype>                                                               \
    {                                                                                                                  \
    public:                                                                                                            \
        OpTile(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_)                                   \
            : OpTileBase<N, Dtype>(sgt_, attribute_, id_)                                                              \
        {}                                                                                                             \
                                                                                                                       \
    protected:                                                                                                         \
        virtual int eval();                                                                                            \
    };

DEF_OP_TILE_RANK(1)
DEF_OP_TILE_RANK(2)
DEF_OP_TILE_RANK(3)
DEF_OP_TILE_RANK(4)
DEF_OP_TILE_RANK(5)
DEF_OP_TILE_RANK(6)

#undef DEF_OP_TILE_RANK

template <int Rank, TOSA_REF_TYPE Dtype>
class OpTranspose : public GraphNode
{
public:
    OpTranspose(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_);
    virtual ~OpTranspose();

    virtual int checkTensorAttributes();
    virtual int eval();

    using InEigenType  = typename GetEigenType<Dtype>::type;
    using OutEigenType = typename GetEigenType<Dtype>::type;
    using TIn          = Eigen::Tensor<InEigenType, Rank>;
    using TOut         = Eigen::Tensor<OutEigenType, Rank>;

protected:
    Eigen::array<int, Rank> perm_array;
    std::unique_ptr<TosaTransposeAttribute> attribute;
    TosaReference::TensorTemplate<TIn>* in;
    TosaReference::TensorTemplate<TOut>* out;
};
};    // namespace TosaReference

#endif

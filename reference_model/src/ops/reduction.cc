
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

#include "reduction.h"
#include "dtype_limits.h"
#include "quant_util.h"

using namespace TosaReference;
using namespace Eigen;
using namespace tosa;

template <int Rank, TOSA_REF_TYPE Dtype>
ReduceNode<Rank, Dtype>::ReduceNode(SubgraphTraverser* sgt_, const Op& op_, TosaAttributeBase* attribute_, uint64_t id_)
    : GraphNode(sgt_, op_, id_)
{
    setRequiredOperands(1, 1);
    setRequiredRank(1);
}

template <int Rank, TOSA_REF_TYPE Dtype>
ReduceNode<Rank, Dtype>::~ReduceNode()
{}

template <int Rank, TOSA_REF_TYPE Dtype>
int ReduceNode<Rank, Dtype>::checkTensorAttributes()
{
    // Check Tosa Level
    auto tosa_level = g_func_config.tosa_level;
    LEVEL_CHECK(Rank <= tosa_level.MAX_RANK, "Rank should be smaller than or equal to MAX_RANK");

    if (validateRequiredOperands())
        return 1;

    if (validateRequiredRank(inputs[0]) || validateRequiredRank(outputs[0]))
    {
        return 1;
    }

    if (axis() < 0 || axis() >= inputs[0]->getRank())
    {
        printNodeValidationError("ReduceOp: axis must between [0, input_rank - 1]");
        return 1;
    }

    if (inputs[0]->matchRankType(*outputs[0]))
    {
        printNodeValidationError("ReduceOp: Input and output tensor ranks must match");
        return 1;
    }

    if (outputs[0]->getShape()[axis()] != 1)
    {
        printNodeValidationError("ReduceOp: Output tensor shape[axis] needs to be 1.");
        return 1;
    }

    in  = dynamic_cast<TosaReference::TensorTemplate<TIn>*>(inputs[0]);
    out = dynamic_cast<TosaReference::TensorTemplate<TOut>*>(outputs[0]);

    if ((!in) || (!out))
    {
        printNodeValidationError("ReduceOp: Input or output fail to cast to Eigen tensor since rank/type not expected");
        return 1;
    }

    dims[0] = axis();

    return 0;
}

// These 2 reducers are to overcome a bug introduced in Eigen between 3.3.7 and 3.4.0
// The in-built .any and .all operations now fail on an assert in TensorMorphing.h:150
// which seems to be due to incorrect data being passed internally as m_impl
struct AllReducer
{
    static const bool PacketAccess = false;
    void reduce(const bool val, bool* accum)
    {
        *accum = *accum && val;
    }
    bool initialize() const
    {
        return true;
    }
    bool finalize(const bool accum) const
    {
        return accum;
    }
};

struct AnyReducer
{
    static const bool PacketAccess = false;
    void reduce(const bool val, bool* accum)
    {
        *accum = *accum || val;
    }
    bool initialize() const
    {
        return false;
    }
    bool finalize(const bool accum) const
    {
        return accum;
    }
};

template <TOSA_REF_TYPE Dtype, typename T>
struct MaxReducer
{
    MaxReducer(NanPropagationMode nan_mode)
        : _nan_mode(nan_mode)
    {}
    void reduce(const T val, T* accum)
    {
        *accum = applyMax<T>(*accum, val, _nan_mode);
    }
    T initialize() const
    {
        return getApplyMaxPadding<Dtype, T>(_nan_mode);
    }
    T finalize(const T accum) const
    {
        return accum;
    }
    static const bool PacketAccess = false;
    NanPropagationMode _nan_mode;
};

template <TOSA_REF_TYPE Dtype, typename T>
struct MinReducer
{
    MinReducer(NanPropagationMode nan_mode)
        : _nan_mode(nan_mode)
    {}
    void reduce(const T val, T* accum)
    {
        *accum = applyMin<T>(*accum, val, _nan_mode);
    }
    T initialize() const
    {
        return getApplyMinPadding<Dtype, T>(_nan_mode);
    }
    T finalize(const T accum) const
    {
        return accum;
    }
    static const bool PacketAccess = false;
    NanPropagationMode _nan_mode;
};

template <int Rank, TOSA_REF_TYPE Dtype>
int OpReduceAll<Rank, Dtype>::eval()
{
    this->out->getTensor() =
        this->in->getTensor().reduce(this->dims, AllReducer()).reshape(this->out->getTensor().dimensions());

    return GraphNode::eval();
}

template <int Rank, TOSA_REF_TYPE Dtype>
int OpReduceAny<Rank, Dtype>::eval()
{
    this->out->getTensor() =
        this->in->getTensor().reduce(this->dims, AnyReducer()).reshape(this->out->getTensor().dimensions());

    return GraphNode::eval();
}

template <int Rank, TOSA_REF_TYPE Dtype>
int OpReduceMax<Rank, Dtype>::checkTensorAttributes()
{
    if (ReduceNode<Rank, Dtype>::checkTensorAttributes())
    {
        return 1;
    }
    if (GraphNode::validateNanMode(attribute->nan_mode()))
    {
        return 1;
    }
    return 0;
}

template <int Rank, TOSA_REF_TYPE Dtype>
int OpReduceMax<Rank, Dtype>::eval()
{
    if constexpr (Dtype == TOSA_REF_TYPE_BF16 || Dtype == TOSA_REF_TYPE_FP16 || Dtype == TOSA_REF_TYPE_FP32)
    {
        this->out->getTensor() = this->in->getTensor()
                                     .reduce(this->dims, MaxReducer<Dtype, float>(this->attribute->nan_mode()))
                                     .reshape(this->out->getTensor().dimensions());
    }
    else if constexpr (Dtype == TOSA_REF_TYPE_FP64)
    {
        this->out->getTensor() = this->in->getTensor()
                                     .reduce(this->dims, MaxReducer<Dtype, double>(this->attribute->nan_mode()))
                                     .reshape(this->out->getTensor().dimensions());
    }
    else if constexpr (Dtype == TOSA_REF_TYPE_INT8 || Dtype == TOSA_REF_TYPE_INT16 || Dtype == TOSA_REF_TYPE_INT32)
    {
        this->out->getTensor() = this->in->getTensor().maximum(this->dims).reshape(this->out->getTensor().dimensions());
    }
    else
    {
        ERROR_IF(true, "unsupported TOSA_REF_TYPE %s", EnumNameTOSAREFTYPE(Dtype));
    }

    return GraphNode::eval();
}

template <int Rank, TOSA_REF_TYPE Dtype>
int OpReduceMin<Rank, Dtype>::checkTensorAttributes()
{
    if (ReduceNode<Rank, Dtype>::checkTensorAttributes())
    {
        return 1;
    }
    if (GraphNode::validateNanMode(attribute->nan_mode()))
    {
        return 1;
    }
    return 0;
}

template <int Rank, TOSA_REF_TYPE Dtype>
int OpReduceMin<Rank, Dtype>::eval()
{
    if constexpr (Dtype == TOSA_REF_TYPE_BF16 || Dtype == TOSA_REF_TYPE_FP16 || Dtype == TOSA_REF_TYPE_FP32)
    {
        this->out->getTensor() = this->in->getTensor()
                                     .reduce(this->dims, MinReducer<Dtype, float>(this->attribute->nan_mode()))
                                     .reshape(this->out->getTensor().dimensions());
    }
    else if constexpr (Dtype == TOSA_REF_TYPE_FP64)
    {
        this->out->getTensor() = this->in->getTensor()
                                     .reduce(this->dims, MinReducer<Dtype, double>(this->attribute->nan_mode()))
                                     .reshape(this->out->getTensor().dimensions());
    }
    else if constexpr (Dtype == TOSA_REF_TYPE_INT8 || Dtype == TOSA_REF_TYPE_INT16 || Dtype == TOSA_REF_TYPE_INT32)
    {
        this->out->getTensor() = this->in->getTensor().minimum(this->dims).reshape(this->out->getTensor().dimensions());
    }
    else
    {
        ERROR_IF(true, "unsupported TOSA_REF_TYPE %s", EnumNameTOSAREFTYPE(Dtype));
    }

    return GraphNode::eval();
}

template <int Rank, TOSA_REF_TYPE Dtype>
int OpReduceProduct<Rank, Dtype>::eval()
{
    switch (Dtype)
    {
        case TOSA_REF_TYPE_FP16:
        case TOSA_REF_TYPE_BF16:
            this->out->getTensor() = this->in->getTensor()
                                         .prod(this->dims)
                                         .reshape(this->out->getTensor().dimensions())
                                         .unaryExpr([](float f) { return fpTrunc<Dtype>(f); });
            break;
        case TOSA_REF_TYPE_FP32:
            this->out->getTensor() =
                this->in->getTensor().prod(this->dims).reshape(this->out->getTensor().dimensions());
            break;
        default:
            ERROR_IF(true, "unsupported TOSA_REF_TYPE %s", EnumNameTOSAREFTYPE(Dtype));
    }

    return GraphNode::eval();
}

struct ProductDoubleReducer
{
    static const bool PacketAccess = false;
    void reduce(const double val, double* accum)
    {
        *accum *= val;
    }
    double initialize() const
    {
        return 1.0;
    }
    double finalize(const double accum) const
    {
        return accum;
    }
};

template <int Rank, TOSA_REF_TYPE Dtype>
int OpReduceProductDouble<Rank, Dtype>::eval()
{
    switch (Dtype)
    {
        case TOSA_REF_TYPE_FP64:
            this->out->getTensor() = this->in->getTensor()
                                         .reduce(this->dims, ProductDoubleReducer())
                                         .reshape(this->out->getTensor().dimensions());
            break;
        default:
            ERROR_IF(true, "unsupported TOSA_REF_TYPE %s", EnumNameTOSAREFTYPE(Dtype));
    }

    return GraphNode::eval();
}

template <int Rank, TOSA_REF_TYPE Dtype>
int OpReduceSum<Rank, Dtype>::eval()
{
    switch (Dtype)
    {
        case TOSA_REF_TYPE_FP16:
        case TOSA_REF_TYPE_BF16:
            this->out->getTensor() = this->in->getTensor()
                                         .sum(this->dims)
                                         .reshape(this->out->getTensor().dimensions())
                                         .unaryExpr([](float f) { return fpTrunc<Dtype>(f); });
            break;
        case TOSA_REF_TYPE_FP32:
        case TOSA_REF_TYPE_INT32:
            this->out->getTensor() = this->in->getTensor().sum(this->dims).reshape(this->out->getTensor().dimensions());
            break;
        default:
            ERROR_IF(true, "unsupported TOSA_REF_TYPE %s", EnumNameTOSAREFTYPE(Dtype));
    }

    return GraphNode::eval();
}

struct SumRequiresReducer
{
    static const bool PacketAccess = false;
    SumRequiresReducer(SubgraphTraverser* parent_sgt)
        : parent_sgt(parent_sgt)
    {}
    void reduce(const int32_t val, int32_t* accum)
    {
        int64_t res_in_64     = static_cast<int64_t>(*accum) + val;
        int64_t i32_max_in_64 = static_cast<int64_t>(std::numeric_limits<int32_t>::max());
        int64_t i32_min_in_64 = static_cast<int64_t>(std::numeric_limits<int32_t>::min());
        REQUIRE(res_in_64 <= i32_max_in_64 && res_in_64 >= i32_min_in_64, "OpReduceSum: result not in i32 range");
        *accum = static_cast<int32_t>(res_in_64);
    }
    int32_t initialize() const
    {
        return 0;
    }
    int32_t finalize(const int32_t accum) const
    {
        return accum;
    }

private:
    SubgraphTraverser* parent_sgt;
};

template <int Rank, TOSA_REF_TYPE Dtype>
int OpReduceSumInt<Rank, Dtype>::eval()
{
    this->out->getTensor() = this->in->getTensor()
                                 .reduce(this->dims, SumRequiresReducer(this->parent_sgt))
                                 .reshape(this->out->getTensor().dimensions());

    return GraphNode::eval();
}

struct SumDoubleReducer
{
    static const bool PacketAccess = false;
    void reduce(const double val, double* accum)
    {
        *accum += val;
    }
    double initialize() const
    {
        return 0.0;
    }
    double finalize(const double accum) const
    {
        return accum;
    }
};

template <int Rank, TOSA_REF_TYPE Dtype>
int OpReduceSumDouble<Rank, Dtype>::eval()
{
    typename ReduceNode<Rank, Dtype>::TIn in_val = this->in->getTensor();
    if (g_func_config.abs_mode)
    {
        // in abs_mode: take abs values of in value
        in_val = in_val.abs();
    }
    switch (Dtype)
    {
        case TOSA_REF_TYPE_FP64:
            this->out->getTensor() =
                in_val.reduce(this->dims, SumDoubleReducer()).reshape(this->out->getTensor().dimensions());
            break;
        default:
            ERROR_IF(true, "unsupported TOSA_REF_TYPE %s", EnumNameTOSAREFTYPE(Dtype));
    }

    return GraphNode::eval();
}

// template explicit instantiation
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceAll, BOOL);

DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceAny, BOOL);

DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceMax, FP16);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceMax, BF16);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceMax, FP32);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceMax, INT8);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceMax, INT16);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceMax, INT32);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceMax, FP64);

DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceMin, FP16);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceMin, BF16);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceMin, FP32);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceMin, INT8);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceMin, INT16);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceMin, INT32);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceMin, FP64);

DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceProduct, FP16);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceProduct, BF16);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceProduct, FP32);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceProductDouble, FP64);

DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceSum, FP16);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceSum, BF16);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceSum, FP32);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceSumDouble, FP64);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceSumInt, INT32);

DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(ReduceNode, BOOL);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(ReduceNode, INT8);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(ReduceNode, INT16);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(ReduceNode, INT32);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(ReduceNode, FP16);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(ReduceNode, BF16);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(ReduceNode, FP32);
DEF_INSTANTIATE_RANK1_6_ONE_RANK_ONE_TYPE(ReduceNode, FP64);

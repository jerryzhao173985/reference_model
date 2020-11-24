
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

#include "op_factory.h"
#include "activation_funcs.h"
#include "comparison.h"
#include "control_flow.h"
#include "custom.h"
#include "data_layout.h"
#include "data_nodes.h"
#include "ewise_binary.h"
#include "ewise_ternary.h"
#include "ewise_unary.h"
#include "image.h"
#include "reduction.h"
#include "scatter_gather.h"
#include "tensor_ops.h"
#include "type_conversion.h"

using namespace TosaReference;
using namespace tosa;

GraphNode* OpFactory::newOp(TosaSerializationHandler* tsh,
                            Op opType,
                            TosaAttributeBase* attribute,
                            TosaQuantInfoBase* qinfo,
                            uint64_t id,
                            DType inputDType,
                            int inputRank,
                            DType outputDType,
                            int outputRank,
                            DType weightDType,
                            int weightRank)
{
    switch (opType)
    {
        // tensor_ops
        case Op_ARGMAX:
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpArgMax, FLOAT);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpArgMax, AINT8);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpArgMax, INT16);
            break;
        case Op_AVG_POOL2D:
            DEF_FACTORY_ONE_TYPE(OpAvgPool2d, FLOAT);
            DEF_FACTORY_ONE_TYPE(OpAvgPool2d, AINT8);
            DEF_FACTORY_ONE_TYPE(OpAvgPool2d, INT16);
            break;
        case Op_CONV2D:
            DEF_FACTORY_TWO_TYPE(OpConv2d, FLOAT, FLOAT);
            DEF_FACTORY_TWO_TYPE(OpConv2d, AINT8, INT4);
            DEF_FACTORY_TWO_TYPE(OpConv2d, AINT8, INT8);
            DEF_FACTORY_TWO_TYPE(OpConv2d, AINT8, AINT8);
            DEF_FACTORY_TWO_TYPE(OpConv2d, INT16, INT8);
            break;
        case Op_DEPTHWISE_CONV2D:
            DEF_FACTORY_TWO_TYPE(OpDepthwiseConv2d, FLOAT, FLOAT);
            DEF_FACTORY_TWO_TYPE(OpDepthwiseConv2d, AINT8, INT4);
            DEF_FACTORY_TWO_TYPE(OpDepthwiseConv2d, AINT8, INT8);
            DEF_FACTORY_TWO_TYPE(OpDepthwiseConv2d, AINT8, AINT8);
            DEF_FACTORY_TWO_TYPE(OpDepthwiseConv2d, INT16, INT8);
            break;
        case Op_FULLY_CONNECTED:
            DEF_FACTORY_TWO_TYPE(OpFullyConnected, FLOAT, FLOAT);
            DEF_FACTORY_TWO_TYPE(OpFullyConnected, AINT8, INT4);
            DEF_FACTORY_TWO_TYPE(OpFullyConnected, AINT8, INT8);
            DEF_FACTORY_TWO_TYPE(OpFullyConnected, AINT8, AINT8);
            DEF_FACTORY_TWO_TYPE(OpFullyConnected, INT16, INT8);
            break;
        case Op_MATMUL:
            DEF_FACTORY_ONE_TYPE(OpMatMul, FLOAT);
            DEF_FACTORY_ONE_TYPE(OpMatMul, AINT8);
            DEF_FACTORY_ONE_TYPE(OpMatMul, INT16);
            break;
        case Op_MAX_POOL2D:
            DEF_FACTORY_ONE_TYPE(OpMaxPool2d, FLOAT);
            DEF_FACTORY_ONE_TYPE(OpMaxPool2d, AINT8);
            DEF_FACTORY_ONE_TYPE(OpMaxPool2d, INT16);
            break;
        case Op_TRANSPOSE_CONV2D:
            DEF_FACTORY_TWO_TYPE(OpTransposeConv2d, FLOAT, FLOAT);
            DEF_FACTORY_TWO_TYPE(OpTransposeConv2d, AINT8, INT4);
            DEF_FACTORY_TWO_TYPE(OpTransposeConv2d, AINT8, INT8);
            DEF_FACTORY_TWO_TYPE(OpTransposeConv2d, AINT8, AINT8);
            DEF_FACTORY_TWO_TYPE(OpTransposeConv2d, INT16, INT8);
            break;

        // activation_funcs
        case Op_CLAMP:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpClamp, FLOAT);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpClamp, AINT8);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpClamp, INT16);
            break;
        case Op_RELUN:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpReluN, FLOAT);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpReluN, INT32);
            break;
        case Op_SIGMOID:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpSigmoid, FLOAT);
            break;
        case Op_TANH:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpTanh, FLOAT);
            break;

        // ewise_binary
        case Op_ADD:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpAdd, FLOAT);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpAdd, INT32);
            break;
        case Op_ARITHMETIC_RIGHT_SHIFT:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpArithmeticRightShift, INT8);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpArithmeticRightShift, INT16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpArithmeticRightShift, INT32);
            break;
        case Op_BITWISE_AND:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpBitwiseAnd, AINT8);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpBitwiseAnd, INT16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpBitwiseAnd, INT32);
            break;
        case Op_BITWISE_OR:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpBitwiseOr, AINT8);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpBitwiseOr, INT16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpBitwiseOr, INT32);
            break;
        case Op_BITWISE_XOR:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpBitwiseXor, AINT8);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpBitwiseXor, INT16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpBitwiseXor, INT32);
            break;
        case Op_LOGICAL_AND:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpLogicalAnd, BOOL);
            break;
        case Op_LOGICAL_LEFT_SHIFT:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpLogicalLeftShift, INT8);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpLogicalLeftShift, INT16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpLogicalLeftShift, INT32);
            break;
        case Op_LOGICAL_RIGHT_SHIFT:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpLogicalRightShift, INT8);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpLogicalRightShift, INT16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpLogicalRightShift, INT32);
            break;
        case Op_LOGICAL_OR:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpLogicalOr, BOOL);
            break;
        case Op_LOGICAL_XOR:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpLogicalOr, BOOL);
            break;
        case Op_MAXIMUM:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpMaximum, FLOAT);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpMaximum, INT32);
            break;
        case Op_MINIMUM:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpMinimum, FLOAT);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpMinimum, INT32);
            break;
        case Op_MUL:
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpMul, FLOAT, FLOAT);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpMul, INT8, INT32);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpMul, INT16, INT32);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpMul, INT32, INT32);
            break;
        case Op_POW:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpPow, FLOAT);
            break;
        case Op_SUB:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpSub, FLOAT);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpSub, INT32);
            break;
        case Op_TABLE:
            DEF_FACTORY_ONE_RANK_0_6(OpTable);
            break;

        // ewise_unary
        case Op_ABS:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpAbs, FLOAT);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpAbs, INT32);
            break;
        case Op_BITWISE_NOT:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpBitwiseNot, AINT8);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpBitwiseNot, INT16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpBitwiseNot, INT32);
            break;
        case Op_CEIL:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpCeil, FLOAT);
            break;
        case Op_CLZ:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpClz, INT32);
            break;
        case Op_EXP:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpExp, FLOAT);
            break;
        case Op_FLOOR:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpFloor, FLOAT);
            break;
        case Op_LOG:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpLog, FLOAT);
            break;
        case Op_LOGICAL_NOT:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpLogicalNot, BOOL);
            break;
        case Op_NEGATE:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpNegate, FLOAT);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpNegate, AINT8);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpNegate, INT16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpNegate, INT32);
            break;
        case Op_RECIPROCAL:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpReciprocal, FLOAT);
            break;
        case Op_RSQRT:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpRsqrt, FLOAT);
            break;

        // ewise_ternary
        case Op_SELECT:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpSelect, FLOAT);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpSelect, AINT8);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpSelect, INT8);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpSelect, INT16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpSelect, INT32);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpSelect, BOOL);
            break;

        // comparison
        case Op_EQUAL:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpEqual, FLOAT);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpEqual, INT32);
            break;
        case Op_GREATER:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpGreater, FLOAT);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpGreater, INT32);
            break;
        case Op_GREATER_EQUAL:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpGreaterEqual, FLOAT);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpGreaterEqual, INT32);
            break;

        // reduction
        case Op_REDUCE_ALL:
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceAll, BOOL);
            break;
        case Op_REDUCE_ANY:
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceAny, BOOL);
            break;
        case Op_REDUCE_MAX:
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceMax, FLOAT);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceMax, AINT8);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceMax, INT16);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceMax, INT32);
            break;
        case Op_REDUCE_MIN:
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceMin, FLOAT);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceMin, AINT8);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceMin, INT16);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceMin, INT32);
            break;
        case Op_REDUCE_PRODUCT:
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceProduct, FLOAT);
            break;
        case Op_REDUCE_SUM:
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceSum, FLOAT);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceSum, INT32);
            break;

        // data layout
        case Op_CONCAT:
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpConcat, FLOAT);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpConcat, AINT8);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpConcat, INT8);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpConcat, INT16);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpConcat, INT32);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpConcat, BOOL);
            break;
        case Op_PAD:
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpPad, FLOAT);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpPad, INT32);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpPad, AINT8);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpPad, INT8);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpPad, INT16);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpPad, BOOL);
            break;
        case Op_RESHAPE:
            DEF_FACTORY_RESHAPE(OpReshape, FLOAT);
            DEF_FACTORY_RESHAPE(OpReshape, AINT8);
            DEF_FACTORY_RESHAPE(OpReshape, INT8);
            DEF_FACTORY_RESHAPE(OpReshape, INT16);
            DEF_FACTORY_RESHAPE(OpReshape, INT32);
            DEF_FACTORY_RESHAPE(OpReshape, BOOL);
            break;
        case Op_REVERSE:
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpReverse, FLOAT);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpReverse, AINT8);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpReverse, INT8);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpReverse, INT16);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpReverse, INT32);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpReverse, BOOL);
            break;
        case Op_SLICE:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpSlice, FLOAT);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpSlice, AINT8);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpSlice, INT8);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpSlice, INT16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpSlice, INT32);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpSlice, BOOL);
            break;
        case Op_TILE:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpTile, FLOAT);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpTile, AINT8);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpTile, INT8);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpTile, INT16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpTile, INT32);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpTile, BOOL);
            break;
        case Op_TRANSPOSE:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpTranspose, BOOL);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpTranspose, FLOAT);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpTranspose, AINT8);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpTranspose, INT8);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpTranspose, INT16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpTranspose, INT32);
            break;

        // scatter_gather
        case Op_GATHER:
            DEF_FACTORY_ONE_TYPE(OpGather, AINT8);
            DEF_FACTORY_ONE_TYPE(OpGather, INT16);
            DEF_FACTORY_ONE_TYPE(OpGather, INT32);
            DEF_FACTORY_ONE_TYPE(OpGather, FLOAT);
            break;
        case Op_SCATTER:
            DEF_FACTORY_ONE_TYPE(OpScatter, AINT8);
            DEF_FACTORY_ONE_TYPE(OpScatter, INT16);
            DEF_FACTORY_ONE_TYPE(OpScatter, INT32);
            DEF_FACTORY_ONE_TYPE(OpScatter, FLOAT);
            break;

        // image
        case Op_RESIZE:
            DEF_FACTORY_TWO_TYPE_RESIZE(OpResize, INT8, INT32);
            DEF_FACTORY_TWO_TYPE_RESIZE(OpResize, INT8, INT8);
            DEF_FACTORY_TWO_TYPE_RESIZE(OpResize, INT16, INT48);
            DEF_FACTORY_TWO_TYPE_RESIZE(OpResize, INT16, INT16);
            DEF_FACTORY_TWO_TYPE_RESIZE(OpResize, FLOAT, FLOAT);
            break;

        // data_nodes
        case Op_CONST:
            return new OpConst(id);
        case Op_PLACEHOLDER:
            return new OpPlaceholder(id);
        case Op_IDENTITY:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpIdentity, FLOAT);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpIdentity, INT32);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpIdentity, AINT8);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpIdentity, INT8);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpIdentity, INT16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpIdentity, BOOL);
            break;
        case Op_IDENTITYN:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpIdentityN, FLOAT);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpIdentityN, INT32);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpIdentityN, AINT8);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpIdentityN, INT8);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpIdentityN, INT16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpIdentityN, BOOL);
            break;

        // type_conversion
        case Op_CAST:
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, BOOL, INT8);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, BOOL, INT16);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, BOOL, INT32);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, INT8, BOOL);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, INT8, INT16);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, INT8, INT32);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, INT8, FLOAT);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, INT16, BOOL);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, INT16, INT8);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, INT16, INT32);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, INT16, FLOAT);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, INT32, BOOL);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, INT32, INT8);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, INT32, INT16);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, INT32, FLOAT);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, FLOAT, INT8);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, FLOAT, INT16);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, FLOAT, INT32);
            break;
        case Op_RESCALE:
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, AINT8, AINT8);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, AINT8, INT16);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, AINT8, INT32);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, INT16, AINT8);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, INT16, INT16);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, INT16, INT32);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, INT32, AINT8);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, INT32, INT16);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, INT32, INT32);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, INT48, AINT8);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, INT48, INT16);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, INT48, INT32);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, UINT8, AINT8);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, AINT8, UINT8);
            break;

        // custom
        case Op_CUSTOM:
            return new OpCustom(id);

        // control_flow
        case Op_COND_IF:
            return new OpCondIf(tsh, attribute, id);
        case Op_WHILE_LOOP:
            return new OpWhileLoop(tsh, attribute, id);

        // Ops not recognized
        default:
            goto done;

    }    // End of switch(opType)

done:
    return nullptr;
}

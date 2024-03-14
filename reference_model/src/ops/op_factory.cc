
// Copyright (c) 2020-2024, ARM Limited.
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
#include "shape.h"
#include "tensor_ops.h"
#include "type_conversion.h"

using namespace TosaReference;
using namespace tosa;

GraphNode* OpFactory::newOp(SubgraphTraverser* sgt,
                            TosaSerializationHandler* tsh,
                            Op opType,
                            TosaAttributeBase* attribute,
                            uint64_t id,
                            TOSA_REF_TYPE inputDTYPE,
                            int inputRank,
                            TOSA_REF_TYPE outputDTYPE,
                            int outputRank,
                            TOSA_REF_TYPE weightDTYPE,
                            int weightRank)
{
    switch (opType)
    {
        // tensor_ops
        case Op_ARGMAX:
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpArgMax, FP16);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpArgMax, BF16);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpArgMax, FP32);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpArgMax, INT8);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpArgMax, INT16);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpArgMax, FP64);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpArgMax, FP8E4M3);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpArgMax, FP8E5M2);
            break;
        case Op_AVG_POOL2D:
            DEF_FACTORY_ONE_TYPE_ONE_ACCUM(OpAvgPool2d, Pool, FP16, FP16);
            DEF_FACTORY_ONE_TYPE_ONE_ACCUM(OpAvgPool2d, Pool, FP16, FP32);
            DEF_FACTORY_ONE_TYPE_ONE_ACCUM(OpAvgPool2d, Pool, BF16, FP32);
            DEF_FACTORY_ONE_TYPE_ONE_ACCUM(OpAvgPool2d, Pool, FP32, FP32);
            DEF_FACTORY_ONE_TYPE_ONE_ACCUM(OpAvgPool2d, Pool, INT8, INT32);
            DEF_FACTORY_ONE_TYPE_ONE_ACCUM(OpAvgPool2d, Pool, INT16, INT32);
            DEF_FACTORY_ONE_TYPE_ONE_ACCUM(OpAvgPool2d, Pool, FP64, FP64);
            DEF_FACTORY_ONE_TYPE_ONE_ACCUM(OpAvgPool2d, Pool, FP8E4M3, FP16);
            DEF_FACTORY_ONE_TYPE_ONE_ACCUM(OpAvgPool2d, Pool, FP8E5M2, FP16);
            break;
        case Op_CONV2D:
            // OP, attr_name, in_t, w_t, acc_t, out_t
            DEF_FACTORY_THREE_TYPE_ONE_ACCUM(OpConv2d, Conv, FP16, FP16, FP16, FP16);
            DEF_FACTORY_THREE_TYPE_ONE_ACCUM(OpConv2d, Conv, FP16, FP16, FP32, FP16);
            DEF_FACTORY_THREE_TYPE_ONE_ACCUM(OpConv2d, Conv, BF16, BF16, FP32, BF16);
            DEF_FACTORY_THREE_TYPE_ONE_ACCUM(OpConv2d, Conv, FP32, FP32, FP32, FP32);
            DEF_FACTORY_THREE_TYPE_ONE_ACCUM(OpConv2d, Conv, INT8, INT4, INT32, INT32);
            DEF_FACTORY_THREE_TYPE_ONE_ACCUM(OpConv2d, Conv, INT8, INT8, INT32, INT32);
            DEF_FACTORY_THREE_TYPE_ONE_ACCUM(OpConv2d, Conv, INT16, INT8, INT48, INT48);
            DEF_FACTORY_THREE_TYPE_ONE_ACCUM(OpConv2d, Conv, FP64, FP64, FP64, FP64);
            DEF_FACTORY_THREE_TYPE_ONE_ACCUM(OpConv2d, Conv, FP8E4M3, FP8E4M3, FP16, FP16);
            DEF_FACTORY_THREE_TYPE_ONE_ACCUM(OpConv2d, Conv, FP8E5M2, FP8E5M2, FP16, FP16);
            break;
        case Op_CONV3D:
            // OP, attr_name, in_t, w_t, acc_t, out_t
            DEF_FACTORY_THREE_TYPE_ONE_ACCUM(OpConv3d, Conv, FP16, FP16, FP16, FP16);
            DEF_FACTORY_THREE_TYPE_ONE_ACCUM(OpConv3d, Conv, FP16, FP16, FP32, FP16);
            DEF_FACTORY_THREE_TYPE_ONE_ACCUM(OpConv3d, Conv, BF16, BF16, FP32, BF16);
            DEF_FACTORY_THREE_TYPE_ONE_ACCUM(OpConv3d, Conv, FP32, FP32, FP32, FP32);
            DEF_FACTORY_THREE_TYPE_ONE_ACCUM(OpConv3d, Conv, INT8, INT4, INT32, INT32);
            DEF_FACTORY_THREE_TYPE_ONE_ACCUM(OpConv3d, Conv, INT8, INT8, INT32, INT32);
            DEF_FACTORY_THREE_TYPE_ONE_ACCUM(OpConv3d, Conv, INT16, INT8, INT48, INT48);
            DEF_FACTORY_THREE_TYPE_ONE_ACCUM(OpConv3d, Conv, FP64, FP64, FP64, FP64);
            DEF_FACTORY_THREE_TYPE_ONE_ACCUM(OpConv3d, Conv, FP8E4M3, FP8E4M3, FP16, FP16);
            DEF_FACTORY_THREE_TYPE_ONE_ACCUM(OpConv3d, Conv, FP8E5M2, FP8E5M2, FP16, FP16);
            break;
        case Op_DEPTHWISE_CONV2D:
            // OP, attr_name, in_t, w_t, acc_t, out_t
            DEF_FACTORY_THREE_TYPE_ONE_ACCUM(OpDepthwiseConv2d, Conv, FP16, FP16, FP16, FP16);
            DEF_FACTORY_THREE_TYPE_ONE_ACCUM(OpDepthwiseConv2d, Conv, FP16, FP16, FP32, FP16);
            DEF_FACTORY_THREE_TYPE_ONE_ACCUM(OpDepthwiseConv2d, Conv, BF16, BF16, FP32, BF16);
            DEF_FACTORY_THREE_TYPE_ONE_ACCUM(OpDepthwiseConv2d, Conv, FP32, FP32, FP32, FP32);
            DEF_FACTORY_THREE_TYPE_ONE_ACCUM(OpDepthwiseConv2d, Conv, INT8, INT4, INT32, INT32);
            DEF_FACTORY_THREE_TYPE_ONE_ACCUM(OpDepthwiseConv2d, Conv, INT8, INT8, INT32, INT32);
            DEF_FACTORY_THREE_TYPE_ONE_ACCUM(OpDepthwiseConv2d, Conv, INT16, INT8, INT48, INT48);
            DEF_FACTORY_THREE_TYPE_ONE_ACCUM(OpDepthwiseConv2d, Conv, FP64, FP64, FP64, FP64);
            DEF_FACTORY_THREE_TYPE_ONE_ACCUM(OpDepthwiseConv2d, Conv, FP8E4M3, FP8E4M3, FP16, FP16);
            DEF_FACTORY_THREE_TYPE_ONE_ACCUM(OpDepthwiseConv2d, Conv, FP8E5M2, FP8E5M2, FP16, FP16);
            break;
        case Op_FFT2D:
            DEF_FACTORY_ONE_TYPE(OpFFT2d, FP32);
            DEF_FACTORY_ONE_TYPE(OpFFT2d, FP64);
            break;
        case Op_FULLY_CONNECTED:
            DEF_FACTORY_THREE_TYPE(OpFullyConnected, FP16, FP16, FP16);
            DEF_FACTORY_THREE_TYPE(OpFullyConnected, FP16, FP16, FP32);
            DEF_FACTORY_THREE_TYPE(OpFullyConnected, BF16, BF16, FP32);
            DEF_FACTORY_THREE_TYPE(OpFullyConnected, FP32, FP32, FP32);
            DEF_FACTORY_THREE_TYPE(OpFullyConnected, INT8, INT4, INT32);
            DEF_FACTORY_THREE_TYPE(OpFullyConnected, INT8, INT8, INT32);
            DEF_FACTORY_THREE_TYPE(OpFullyConnected, INT16, INT8, INT48);
            DEF_FACTORY_THREE_TYPE(OpFullyConnected, FP64, FP64, FP64);
            DEF_FACTORY_THREE_TYPE(OpFullyConnected, FP8E4M3, FP8E4M3, FP16);
            DEF_FACTORY_THREE_TYPE(OpFullyConnected, FP8E5M2, FP8E5M2, FP16);
            break;
        case Op_MATMUL:
            DEF_FACTORY_TWO_TYPE_IN_OUT(OpMatMul, FP16, FP16);
            DEF_FACTORY_TWO_TYPE_IN_OUT(OpMatMul, FP16, FP32);
            DEF_FACTORY_TWO_TYPE_IN_OUT(OpMatMul, BF16, FP32);
            DEF_FACTORY_TWO_TYPE_IN_OUT(OpMatMul, FP32, FP32);
            DEF_FACTORY_TWO_TYPE_IN_OUT(OpMatMul, INT8, INT32);
            DEF_FACTORY_TWO_TYPE_IN_OUT(OpMatMul, INT16, INT48);
            DEF_FACTORY_TWO_TYPE_IN_OUT(OpMatMul, FP64, FP64);
            DEF_FACTORY_TWO_TYPE_IN_OUT(OpMatMul, FP8E4M3, FP16);
            DEF_FACTORY_TWO_TYPE_IN_OUT(OpMatMul, FP8E5M2, FP16);
            break;
        case Op_MAX_POOL2D:
            DEF_FACTORY_ONE_TYPE(OpMaxPool2d, FP16);
            DEF_FACTORY_ONE_TYPE(OpMaxPool2d, BF16);
            DEF_FACTORY_ONE_TYPE(OpMaxPool2d, FP32);
            DEF_FACTORY_ONE_TYPE(OpMaxPool2d, INT8);
            DEF_FACTORY_ONE_TYPE(OpMaxPool2d, INT16);
            DEF_FACTORY_ONE_TYPE(OpMaxPool2d, FP64);
            DEF_FACTORY_ONE_TYPE(OpMaxPool2d, FP8E4M3);
            DEF_FACTORY_ONE_TYPE(OpMaxPool2d, FP8E5M2);
            break;
        case Op_RFFT2D:
            DEF_FACTORY_ONE_TYPE(OpRFFT2d, FP32);
            DEF_FACTORY_ONE_TYPE(OpRFFT2d, FP64);
            break;
        case Op_TRANSPOSE_CONV2D:
            // OP, attr_name, in_t, w_t, acc_t, out_t
            DEF_FACTORY_THREE_TYPE_ONE_ACCUM(OpTransposeConv2d, TransposeConv, FP16, FP16, FP16, FP16);
            DEF_FACTORY_THREE_TYPE_ONE_ACCUM(OpTransposeConv2d, TransposeConv, FP16, FP16, FP32, FP16);
            DEF_FACTORY_THREE_TYPE_ONE_ACCUM(OpTransposeConv2d, TransposeConv, BF16, BF16, FP32, BF16);
            DEF_FACTORY_THREE_TYPE_ONE_ACCUM(OpTransposeConv2d, TransposeConv, FP32, FP32, FP32, FP32);
            DEF_FACTORY_THREE_TYPE_ONE_ACCUM(OpTransposeConv2d, TransposeConv, INT8, INT4, INT32, INT32);
            DEF_FACTORY_THREE_TYPE_ONE_ACCUM(OpTransposeConv2d, TransposeConv, INT8, INT8, INT32, INT32);
            DEF_FACTORY_THREE_TYPE_ONE_ACCUM(OpTransposeConv2d, TransposeConv, INT16, INT8, INT48, INT48);
            DEF_FACTORY_THREE_TYPE_ONE_ACCUM(OpTransposeConv2d, TransposeConv, FP64, FP64, FP64, FP64);
            DEF_FACTORY_THREE_TYPE_ONE_ACCUM(OpTransposeConv2d, TransposeConv, FP8E4M3, FP8E4M3, FP16, FP16);
            DEF_FACTORY_THREE_TYPE_ONE_ACCUM(OpTransposeConv2d, TransposeConv, FP8E5M2, FP8E5M2, FP16, FP16);
            break;

        // activation_funcs
        case Op_CLAMP:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpClamp, FP16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpClamp, BF16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpClamp, FP32);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpClamp, INT8);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpClamp, INT16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpClamp, FP64);
            break;
        case Op_SIGMOID:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpSigmoid, FP16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpSigmoid, BF16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpSigmoid, FP32);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpSigmoid, FP64);
            break;
        case Op_TANH:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpTanh, FP16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpTanh, BF16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpTanh, FP32);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpTanh, FP64);
            break;
        case Op_ERF:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpErf, FP16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpErf, BF16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpErf, FP32);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpErf, FP64);
            break;

        // ewise_binary
        case Op_ADD:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpAdd, FP16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpAdd, BF16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpAdd, FP32);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpAdd, INT32);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpAdd, FP64);
            break;
        case Op_ARITHMETIC_RIGHT_SHIFT:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpArithmeticRightShift, INT8);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpArithmeticRightShift, INT16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpArithmeticRightShift, INT32);
            break;
        case Op_BITWISE_AND:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpBitwiseAnd, INT8);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpBitwiseAnd, INT16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpBitwiseAnd, INT32);
            break;
        case Op_BITWISE_OR:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpBitwiseOr, INT8);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpBitwiseOr, INT16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpBitwiseOr, INT32);
            break;
        case Op_BITWISE_XOR:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpBitwiseXor, INT8);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpBitwiseXor, INT16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpBitwiseXor, INT32);
            break;
        case Op_INTDIV:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpIntdiv, INT32);
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
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpLogicalXor, BOOL);
            break;
        case Op_MAXIMUM:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpMaximum, FP16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpMaximum, BF16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpMaximum, FP32);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpMaximum, INT32);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpMaximum, FP64);
            break;
        case Op_MINIMUM:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpMinimum, FP16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpMinimum, BF16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpMinimum, FP32);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpMinimum, INT32);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpMinimum, FP64);
            break;
        case Op_MUL:
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpMul, FP16, FP16);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpMul, BF16, BF16);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpMul, FP32, FP32);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpMul, INT8, INT32);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpMul, INT16, INT32);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpMul, INT32, INT32);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpMul, FP64, FP64);
            break;
        case Op_POW:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpPow, FP16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpPow, BF16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpPow, FP32);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpPow, FP64);
            break;
        case Op_SUB:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpSub, FP16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpSub, BF16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpSub, FP32);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpSub, INT32);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpSub, FP64);
            break;
        case Op_TABLE:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpTable, INT8);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpTable, INT16);
            break;

        // ewise_unary
        case Op_ABS:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpAbs, FP16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpAbs, BF16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpAbs, FP32);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpAbs, INT32);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpAbs, FP64);
            break;
        case Op_BITWISE_NOT:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpBitwiseNot, INT8);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpBitwiseNot, INT16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpBitwiseNot, INT32);
            break;
        case Op_CEIL:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpCeil, FP16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpCeil, BF16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpCeil, FP32);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpCeil, FP64);
            break;
        case Op_CLZ:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpClz, INT32);
            break;
        case Op_COS:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpCos, FP16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpCos, BF16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpCos, FP32);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpCos, FP64);
            break;
        case Op_EXP:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpExp, FP16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpExp, BF16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpExp, FP32);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpExp, FP64);
            break;
        case Op_FLOOR:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpFloor, FP16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpFloor, BF16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpFloor, FP32);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpFloor, FP64);
            break;
        case Op_LOG:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpLog, FP16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpLog, BF16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpLog, FP32);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpLog, FP64);
            break;
        case Op_LOGICAL_NOT:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpLogicalNot, BOOL);
            break;
        case Op_NEGATE:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpNegate, FP16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpNegate, BF16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpNegate, FP32);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpNegate, INT8);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpNegate, INT16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpNegate, INT32);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpNegate, FP64);
            break;
        case Op_RECIPROCAL:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpReciprocal, FP16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpReciprocal, BF16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpReciprocal, FP32);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpReciprocal, FP64);
            break;
        case Op_RSQRT:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpRsqrt, FP16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpRsqrt, BF16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpRsqrt, FP32);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpRsqrt, FP64);
            break;
        case Op_SIN:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpSin, FP16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpSin, BF16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpSin, FP32);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpSin, FP64);
            break;

        // ewise_ternary
        case Op_SELECT:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpSelect, FP16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpSelect, BF16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpSelect, FP32);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpSelect, INT8);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpSelect, INT16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpSelect, INT32);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpSelect, BOOL);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpSelect, FP64);
            break;

        // comparison
        case Op_EQUAL:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpEqual, FP16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpEqual, BF16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpEqual, FP32);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpEqual, INT32);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpEqual, FP64);
            break;
        case Op_GREATER:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpGreater, FP16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpGreater, BF16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpGreater, FP32);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpGreater, INT32);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpGreater, FP64);
            break;
        case Op_GREATER_EQUAL:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpGreaterEqual, FP16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpGreaterEqual, BF16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpGreaterEqual, FP32);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpGreaterEqual, INT32);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpGreaterEqual, FP64);
            break;

        // reduction
        case Op_REDUCE_ALL:
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceAll, BOOL);
            break;
        case Op_REDUCE_ANY:
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceAny, BOOL);
            break;
        case Op_REDUCE_MAX:
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceMax, FP16);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceMax, BF16);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceMax, FP32);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceMax, INT8);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceMax, INT16);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceMax, INT32);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceMax, FP64);
            break;
        case Op_REDUCE_MIN:
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceMin, FP16);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceMin, BF16);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceMin, FP32);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceMin, INT8);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceMin, INT16);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceMin, INT32);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceMin, FP64);
            break;
        case Op_REDUCE_PRODUCT:
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceProduct, FP16);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceProduct, BF16);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceProduct, FP32);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceProductDouble, FP64);
            break;
        case Op_REDUCE_SUM:
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceSum, FP16);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceSum, BF16);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceSum, FP32);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceSumDouble, FP64);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpReduceSumInt, INT32);
            break;

        // data layout
        case Op_CONCAT:
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpConcat, FP16);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpConcat, BF16);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpConcat, FP32);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpConcat, INT8);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpConcat, INT16);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpConcat, INT32);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpConcat, BOOL);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpConcat, FP64);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpConcat, FP8E4M3);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpConcat, FP8E5M2);
            break;
        case Op_PAD:
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpPad, FP16);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpPad, BF16);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpPad, FP32);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpPad, INT32);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpPad, INT8);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpPad, INT16);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpPad, BOOL);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpPad, FP64);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpPad, FP8E4M3);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpPad, FP8E5M2);
            break;
        case Op_DIM:
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpDim, FP16);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpDim, BF16);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpDim, FP32);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpDim, INT32);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpDim, INT8);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpDim, INT16);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpDim, BOOL);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpDim, FP8E4M3);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpDim, FP8E5M2);
            break;
        case Op_RESHAPE:
            DEF_FACTORY_RESHAPE(OpReshape, FP16);
            DEF_FACTORY_RESHAPE(OpReshape, BF16);
            DEF_FACTORY_RESHAPE(OpReshape, FP32);
            DEF_FACTORY_RESHAPE(OpReshape, INT8);
            DEF_FACTORY_RESHAPE(OpReshape, INT16);
            DEF_FACTORY_RESHAPE(OpReshape, INT32);
            DEF_FACTORY_RESHAPE(OpReshape, BOOL);
            DEF_FACTORY_RESHAPE(OpReshape, FP64);
            DEF_FACTORY_RESHAPE(OpReshape, FP8E4M3);
            DEF_FACTORY_RESHAPE(OpReshape, FP8E5M2);
            break;
        case Op_REVERSE:
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpReverse, FP16);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpReverse, BF16);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpReverse, FP32);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpReverse, INT8);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpReverse, INT16);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpReverse, INT32);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpReverse, BOOL);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpReverse, FP64);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpReverse, FP8E4M3);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpReverse, FP8E5M2);
            break;
        case Op_SLICE:
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpSlice, FP16);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpSlice, BF16);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpSlice, FP32);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpSlice, INT8);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpSlice, INT16);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpSlice, INT32);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpSlice, BOOL);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpSlice, FP64);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpSlice, FP8E4M3);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpSlice, FP8E5M2);
            break;
        case Op_TILE:
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpTile, FP16);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpTile, BF16);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpTile, FP32);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpTile, INT8);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpTile, INT16);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpTile, INT32);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpTile, BOOL);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpTile, FP64);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpTile, FP8E4M3);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpTile, FP8E5M2);
            break;
        case Op_TRANSPOSE:
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpTranspose, BOOL);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpTranspose, FP16);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpTranspose, BF16);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpTranspose, FP32);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpTranspose, INT8);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpTranspose, INT16);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpTranspose, INT32);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpTranspose, FP64);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpTranspose, FP8E4M3);
            DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OpTranspose, FP8E5M2);
            break;

        // scatter_gather
        case Op_GATHER:
            DEF_FACTORY_ONE_TYPE(OpGather, INT8);
            DEF_FACTORY_ONE_TYPE(OpGather, INT16);
            DEF_FACTORY_ONE_TYPE(OpGather, INT32);
            DEF_FACTORY_ONE_TYPE(OpGather, FP16);
            DEF_FACTORY_ONE_TYPE(OpGather, BF16);
            DEF_FACTORY_ONE_TYPE(OpGather, FP32);
            DEF_FACTORY_ONE_TYPE(OpGather, FP64);
            DEF_FACTORY_ONE_TYPE(OpGather, FP8E4M3);
            DEF_FACTORY_ONE_TYPE(OpGather, FP8E5M2);
            break;
        case Op_SCATTER:
            DEF_FACTORY_ONE_TYPE(OpScatter, INT8);
            DEF_FACTORY_ONE_TYPE(OpScatter, INT16);
            DEF_FACTORY_ONE_TYPE(OpScatter, INT32);
            DEF_FACTORY_ONE_TYPE(OpScatter, FP16);
            DEF_FACTORY_ONE_TYPE(OpScatter, BF16);
            DEF_FACTORY_ONE_TYPE(OpScatter, FP32);
            DEF_FACTORY_ONE_TYPE(OpScatter, FP64);
            DEF_FACTORY_ONE_TYPE(OpScatter, FP8E4M3);
            DEF_FACTORY_ONE_TYPE(OpScatter, FP8E5M2);
            break;

        // image
        case Op_RESIZE:
            DEF_FACTORY_TWO_TYPE_RESIZE_INT16(OpResize, INT8, INT32);
            DEF_FACTORY_TWO_TYPE_RESIZE_INT16(OpResize, INT8, INT8);
            DEF_FACTORY_TWO_TYPE_RESIZE_INT16(OpResize, INT16, INT48);
            DEF_FACTORY_TWO_TYPE_RESIZE_INT16(OpResize, INT16, INT16);
            DEF_FACTORY_TWO_TYPE_RESIZE_FP16(OpResize, FP16, FP16);
            DEF_FACTORY_TWO_TYPE_RESIZE_BF16(OpResize, BF16, BF16);
            DEF_FACTORY_TWO_TYPE_RESIZE_FP32(OpResize, FP32, FP32);
            DEF_FACTORY_TWO_TYPE_RESIZE_FP64(OpResize, FP64, FP64);
            break;

        // data_nodes
        case Op_CONST:
            return new OpConst(sgt, id);
        case Op_IDENTITY:
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpIdentity, FP16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpIdentity, BF16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpIdentity, FP32);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpIdentity, INT32);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpIdentity, INT4);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpIdentity, INT8);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpIdentity, INT16);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpIdentity, INT48);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpIdentity, BOOL);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpIdentity, FP64);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpIdentity, FP8E4M3);
            DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OpIdentity, FP8E5M2);
            break;

        // type_conversion
        case Op_CAST:
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, BOOL, INT8);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, BOOL, INT16);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, BOOL, INT32);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, INT8, BOOL);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, INT8, INT16);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, INT8, INT32);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, INT8, FP16);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, INT8, BF16);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, INT8, FP32);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, INT16, BOOL);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, INT16, INT8);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, INT16, INT32);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, INT16, FP16);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, INT16, BF16);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, INT16, FP32);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, INT32, BOOL);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, INT32, INT8);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, INT32, INT16);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, INT32, FP16);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, INT32, BF16);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, INT32, FP32);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, FP16, INT8);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, FP16, INT16);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, FP16, INT32);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, FP16, FP32);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, BF16, INT8);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, BF16, INT16);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, BF16, INT32);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, BF16, FP32);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, FP32, INT8);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, FP32, INT16);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, FP32, INT32);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, FP32, FP16);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, FP32, BF16);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, FP64, INT8);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, FP64, INT16);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, FP64, INT32);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, FP64, FP64);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, INT8, FP64);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, INT16, FP64);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, INT32, FP64);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, BF16, FP8E4M3);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, BF16, FP8E5M2);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, FP8E4M3, FP16);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, FP8E4M3, BF16);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, FP8E4M3, FP32);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, FP8E5M2, FP16);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, FP8E5M2, BF16);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, FP8E5M2, FP32);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, FP16, FP8E4M3);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, FP16, FP8E5M2);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, FP32, FP8E4M3);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpCast, FP32, FP8E5M2);
            break;
        case Op_RESCALE:
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, INT8, INT8);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, INT8, INT16);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, INT8, INT32);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, INT16, INT8);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, INT16, INT16);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, INT16, INT32);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, INT32, INT8);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, INT32, INT16);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, INT32, INT32);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, INT48, INT8);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, INT48, INT16);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, INT48, INT32);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, UINT8, INT8);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, UINT8, INT16);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, UINT16, INT16);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, INT8, UINT8);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, INT16, UINT8);
            DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OpRescale, INT16, UINT16);
            break;

        // custom
        case Op_CUSTOM:
            return new OpCustom(sgt, attribute, id);

        // control_flow
        case Op_COND_IF:
            return new OpCondIf(sgt, tsh, attribute, id);
        case Op_WHILE_LOOP:
            return new OpWhileLoop(sgt, tsh, attribute, id);

        case Op_CONST_SHAPE:
            return new OpConstShape(sgt, id);
        case Op_CONCAT_SHAPE:
            return new OpConcatShape(sgt, id);
        case Op_ADD_SHAPE:
            return new OpAddShape(sgt, id);
        case Op_SUB_SHAPE:
            return new OpSubShape(sgt, id);
        case Op_MUL_SHAPE:
            return new OpMulShape(sgt, id);
        case Op_DIV_SHAPE:
            return new OpDivShape(sgt, id);

        // Ops not recognized
        default:
            goto done;

    }    // End of switch(opType)

done:
    return nullptr;
}

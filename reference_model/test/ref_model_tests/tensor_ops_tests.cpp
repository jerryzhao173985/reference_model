// Copyright (c) 2024-2025 ARM Limited.
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

#include "cfloat.h"
#include "dtype.h"
#include "dtype_limits.h"
#include "half.hpp"

// Include this last because it redefines REQUIRE
#include "test_utils.h"

template <typename IN_TYPE>
void testArgmaxFpSpecial(bool propagate)
{
    constexpr DType inDtype           = NativeType2DType<IN_TYPE>();
    constexpr TOSA_REF_TYPE inReftype = DType2RefType(inDtype);

    const int TEST_ROWS    = 2;
    const int TEST_COLUMNS = 9;

    RefModelTestBuilder tb{};
    tb.addInput({ TEST_ROWS, TEST_COLUMNS }, inDtype);
    tb.addOutput({ TEST_COLUMNS }, DType_INT32);

    TosaArgMaxAttribute attr{ /* axis */ 0, propagate ? NanPropagationMode_PROPAGATE : NanPropagationMode_IGNORE };
    tb.addOp(Op_ARGMAX, Attribute_ArgMaxAttribute, &attr);

    tb.initializeRunner();

    const IN_TYPE lowest = DtypeLimits<inReftype>::lowest;
    const IN_TYPE nan    = DtypeLimits<inReftype>::quiet_NaN;
    const IN_TYPE inf    = DtypeLimits<inReftype>::has_infinity ? DtypeLimits<inReftype>::infinity : nan;
    const bool hasInf    = DtypeLimits<inReftype>::has_infinity;

    // NOTE: Eigen stores Tensors in column-major order but we initialise them row-wise
    std::vector<IN_TYPE> inVals = { // Row 0
                                    IN_TYPE(0), lowest, inf, -inf, -inf, -inf, lowest, nan, nan,
                                    // Row 1
                                    IN_TYPE(2), lowest, inf, -inf, inf, lowest, IN_TYPE(0), IN_TYPE(0), nan
    };

    // (-inf, lowest) in fp8e4m3 is a complicated case because it becomes (NaN, lowest)
    // And so in that case the result depends on propagation
    std::vector<int32_t> expectedOut = {
        1,                                   // 0, 2
        0,                                   // lowest, lowest
        0,                                   // inf, inf
        0,                                   // -inf, -inf
        hasInf ? 1 : 0,                      // -inf, inf
        hasInf ? 1 : (propagate ? 0 : 1),    // -inf, lowest
        1,                                   // lowest, 0
        propagate ? 0 : 1,                   // nan, 0
        0                                    // nan, nan
    };
    tb.setInput(inVals);

    REQUIRE(tb.run() == GraphStatus::TOSA_VALID);
    std::vector<int32_t> actualOut = tb.getOutput<int32_t>(0, /* size */ expectedOut.size());

    compareOutput<int32_t>(expectedOut, actualOut);
}

void testConv2d(std::vector<int8_t>& inVals, std::vector<int8_t>& weightVals, int8_t inZp, int8_t weightZp)
{
    RefModelTestBuilder tb{};
    constexpr DType inDtype     = DType_INT8;
    constexpr DType weightDtype = DType_INT8;
    constexpr DType outDtype    = DType_INT32;

    const int HEIGHT = 2;
    const int WIDTH  = 2;

    REQUIRE_MESSAGE(inVals.size() == HEIGHT * WIDTH, "Unit test construction error: testConv2d assumes the inVals has ",
                    HEIGHT * WIDTH, " elements");
    REQUIRE_MESSAGE(weightVals.size() == HEIGHT * WIDTH,
                    "Unit test construction error: testConv2d assumes the weightVals has ", HEIGHT * WIDTH,
                    " elements");

    tb.addInput({ 1, HEIGHT, WIDTH, 1 }, inDtype);        // input
    tb.addInput({ 1, HEIGHT, WIDTH, 1 }, weightDtype);    // weight
    tb.addInput({ 1 }, outDtype);                         // bias
    tb.addInput({ 1 }, inDtype);                          // in zp
    tb.addInput({ 1 }, weightDtype);                      // weight zp
    tb.addOutput({ 1, 1, 1, 1 }, outDtype);

    auto attr = TosaConv2dAttribute(/* pad */ { 0, 0, 0, 0 }, /* stride */ { 2, 2 }, /* dilation */ { 1, 1 },
                                    /* local_bound */ true, /* acc_t */ DType_INT32);
    tb.addOp(Op_CONV2D, Attribute_Conv2dAttribute, &attr);

    tb.initializeRunner();

    int32_t expectedOutVal = 0;
    for (size_t i = 0; i < inVals.size(); i++)
    {
        expectedOutVal += (static_cast<int32_t>(inVals[i]) - static_cast<int32_t>(inZp)) *
                          (static_cast<int32_t>(weightVals[i]) - static_cast<int32_t>(weightZp));
    }
    std::vector<int32_t> expectedOut = { expectedOutVal };

    std::vector<int8_t> inZpVals     = { inZp };
    std::vector<int8_t> weightZpVals = { weightZp };
    std::vector<int32_t> biasVals    = { 0 };
    tb.setInput(inVals);
    tb.setInput(weightVals);
    tb.setInput(biasVals);
    tb.setInput(inZpVals);
    tb.setInput(weightZpVals);

    REQUIRE(tb.run() == GraphStatus::TOSA_VALID);
    std::vector<int32_t> actualOut = tb.getOutput<int32_t>(0, /* size */ expectedOut.size());

    compareOutput<int32_t>(expectedOut, actualOut);
}

void testConv2dBias(std::vector<int8_t>& inVals,
                    std::vector<int8_t>& weightVals,
                    std::vector<int32_t>& biasVals,
                    int8_t inZp,
                    int8_t weightZp,
                    std::vector<int32_t>& expectedOut)
{
    RefModelTestBuilder tb{};
    const DType inDtype  = DType_INT8;
    const DType outDtype = DType_INT32;

    const int IN_CHANNELS  = 2;
    const int OUT_CHANNELS = 2;

    tb.addInput({ 1, 1, 1, IN_CHANNELS }, inDtype);                      // input
    tb.addInput({ OUT_CHANNELS, 1, 1, IN_CHANNELS }, inDtype);           // weight
    tb.addInput({ static_cast<int32_t>(biasVals.size()) }, outDtype);    // bias
    tb.addInput({ 1 }, inDtype);                                         // input zp
    tb.addInput({ 1 }, inDtype);                                         // weight zp
    tb.addOutput({ 1, 1, 1, OUT_CHANNELS }, outDtype);

    auto attr = TosaConv2dAttribute(/* pad */ { 0, 0, 0, 0 },
                                    /* stride */ { 1, 1 },
                                    /* dilation */ { 1, 1 },
                                    /* local_bound */ true,
                                    /* acc_type */ DType_INT32);

    tb.addOp(Op_CONV2D, Attribute_Conv2dAttribute, &attr);

    tb.initializeRunner();

    std::vector<int8_t> inZpVals     = { inZp };
    std::vector<int8_t> weightZpVals = { weightZp };

    tb.setInput(inVals);
    tb.setInput(weightVals);
    tb.setInput(biasVals);
    tb.setInput(inZpVals);
    tb.setInput(weightZpVals);

    REQUIRE(tb.run() == GraphStatus::TOSA_VALID);
    std::vector<int32_t> actualOut = tb.getOutput<int32_t>(0, /* size */ expectedOut.size());

    compareOutput<int32_t>(expectedOut, actualOut);
}

void testDepthwiseConv2d(std::vector<int8_t>& inVals, std::vector<int8_t>& weightVals, int8_t inZp, int8_t weightZp)
{
    RefModelTestBuilder tb{};
    constexpr DType inDtype     = DType_INT8;
    constexpr DType weightDtype = DType_INT8;
    constexpr DType outDtype    = DType_INT32;

    const int HEIGHT = 2;
    const int WIDTH  = 2;

    REQUIRE_MESSAGE(inVals.size() == HEIGHT * WIDTH,
                    "Unit test construction error: testDeptwiseConv2d assumes the inVals has ", HEIGHT * WIDTH,
                    " elements");
    REQUIRE_MESSAGE(weightVals.size() == HEIGHT * WIDTH,
                    "Unit test construction error: testDepthConv2d assumes the weightVals has ", HEIGHT * WIDTH,
                    " elements");

    tb.addInput({ 1, HEIGHT, WIDTH, 1 }, inDtype);
    tb.addInput({ HEIGHT, WIDTH, 1, 1 }, weightDtype);
    tb.addInput({ 1 }, outDtype);
    tb.addInput({ 1 }, inDtype);
    tb.addInput({ 1 }, weightDtype);
    tb.addOutput({ 1, 1, 1, 1 }, outDtype);

    auto attr = TosaDepthwiseConv2dAttribute({ 0, 0, 0, 0 }, { 2, 2 }, { 1, 1 }, true, DType_INT32);
    tb.addOp(Op_DEPTHWISE_CONV2D, Attribute_DepthwiseConv2dAttribute, &attr);

    tb.initializeRunner();

    int32_t expectedOutVal = 0;
    for (size_t i = 0; i < inVals.size(); i++)
    {
        expectedOutVal += (static_cast<int32_t>(inVals[i]) - static_cast<int32_t>(inZp)) *
                          (static_cast<int32_t>(weightVals[i]) - static_cast<int32_t>(weightZp));
    }
    std::vector<int32_t> expectedOut = { expectedOutVal };

    std::vector<int8_t> inZpVals     = { inZp };
    std::vector<int8_t> weightZpVals = { weightZp };
    std::vector<int32_t> biasVals    = { 0 };
    tb.setInput(inVals);
    tb.setInput(weightVals);
    tb.setInput(biasVals);
    tb.setInput(inZpVals);
    tb.setInput(weightZpVals);

    REQUIRE(tb.run() == GraphStatus::TOSA_VALID);
    std::vector<int32_t> actualOut = tb.getOutput<int32_t>(0, /* size */ expectedOut.size());

    compareOutput<int32_t>(expectedOut, actualOut);
}

void testDepthwiseConv2dBias(std::vector<int8_t>& inVals,
                             std::vector<int8_t>& weightVals,
                             std::vector<int32_t>& biasVals,
                             int8_t inZp,
                             int8_t weightZp,
                             std::vector<int32_t>& expectedOut)
{
    RefModelTestBuilder tb{};
    const DType inDtype  = DType_INT8;
    const DType outDtype = DType_INT32;

    const int IN_CHANNELS = 2;

    tb.addInput({ 1, 1, 1, IN_CHANNELS }, inDtype);                      // input
    tb.addInput({ 1, 1, IN_CHANNELS, 1 }, inDtype);                      // weight
    tb.addInput({ static_cast<int32_t>(biasVals.size()) }, outDtype);    // bias
    tb.addInput({ 1 }, inDtype);                                         // input zp
    tb.addInput({ 1 }, inDtype);                                         // weight zp
    tb.addOutput({ 1, 1, 1, IN_CHANNELS }, outDtype);

    auto attr = TosaDepthwiseConv2dAttribute(/* pad */ { 0, 0, 0, 0 },
                                             /* stride */ { 1, 1 },
                                             /* dilation */ { 1, 1 },
                                             /* local_bound */ true,
                                             /* acc_type */ DType_INT32);

    tb.addOp(Op_DEPTHWISE_CONV2D, Attribute_DepthwiseConv2dAttribute, &attr);

    tb.initializeRunner();

    std::vector<int8_t> inZpVals     = { inZp };
    std::vector<int8_t> weightZpVals = { weightZp };

    tb.setInput(inVals);
    tb.setInput(weightVals);
    tb.setInput(biasVals);
    tb.setInput(inZpVals);
    tb.setInput(weightZpVals);

    REQUIRE(tb.run() == GraphStatus::TOSA_VALID);
    std::vector<int32_t> actualOut = tb.getOutput<int32_t>(0, /* size */ expectedOut.size());

    compareOutput<int32_t>(expectedOut, actualOut);
}

void testTransposeConv2d(std::vector<int8_t>& inVals, std::vector<int8_t>& weightVals, int8_t inZp, int8_t weightZp)
{
    RefModelTestBuilder tb{};
    constexpr DType inDtype     = DType_INT8;
    constexpr DType weightDtype = DType_INT8;
    constexpr DType outDtype    = DType_INT32;

    const int IN_HEIGHT = 2;
    const int IN_WIDTH  = 2;

    REQUIRE_MESSAGE(inVals.size() == IN_HEIGHT * IN_WIDTH,
                    "Unit test construction error: testTransposeConv2d assumes the inVals has ", IN_HEIGHT * IN_WIDTH,
                    " elements");
    REQUIRE_MESSAGE(weightVals.size() == IN_HEIGHT * IN_WIDTH,
                    "Unit test construction error: testTrasposeConv2d assumes the weightVals has ",
                    IN_HEIGHT * IN_WIDTH, " elements");

    tb.addInput({ 1, IN_HEIGHT, IN_WIDTH, 1 }, inDtype);        // input
    tb.addInput({ 1, IN_HEIGHT, IN_WIDTH, 1 }, weightDtype);    // weight
    tb.addInput({ 1 }, outDtype);                               // bias
    tb.addInput({ 1 }, inDtype);                                // input zp
    tb.addInput({ 1 }, weightDtype);                            // weight zp
    tb.addOutput({ 1, 3, 3, 1 }, outDtype);

    auto attr = TosaTransposeConv2dAttribute(/* pad */ { 0, 0, 0, 0 },
                                             /* stride */ { 1, 1 },
                                             /* local_bound */ true,
                                             /* acc_type */ DType_INT32);

    // Formula: OH = (IH - 1) * stride_y + out_pad_top + out_pad_bottom + KH
    const int OUT_HEIGHT = IN_HEIGHT * 2 - 1;
    // Formula: OW = (IW - 1) * stride_x + out_pad_left + out_pad_right + KW
    const int OUT_WIDTH = IN_WIDTH * 2 - 1;

    tb.addOp(Op_TRANSPOSE_CONV2D, Attribute_TransposeConv2dAttribute, &attr);

    tb.initializeRunner();

    std::vector<int32_t> expectedOutTensor(9, 0);
    for (int outRow = 0; outRow < OUT_HEIGHT; outRow++)
    {
        for (int outCol = 0; outCol < OUT_WIDTH; outCol++)
        {
            int32_t value = 0;

            for (int kRow = 0; kRow < IN_HEIGHT; kRow++)
            {
                for (int kCol = 0; kCol < IN_WIDTH; kCol++)
                {
                    int inRow = outRow - kRow;
                    int inCol = outCol - kCol;

                    if (inRow >= 0 && inRow < IN_HEIGHT && inCol >= 0 && inCol < IN_WIDTH)
                    {
                        int inIdx     = inRow * IN_HEIGHT + inCol;
                        int weightIdx = kRow * IN_WIDTH + kCol;

                        value += (static_cast<int32_t>(inVals[inIdx]) - static_cast<int32_t>(inZp)) *
                                 (static_cast<int32_t>(weightVals[weightIdx]) - static_cast<int32_t>(weightZp));
                    }
                }
            }
            expectedOutTensor[outRow * OUT_WIDTH + outCol] = value;
        }
    }
    std::vector<int32_t> expectedOut = { expectedOutTensor };

    std::vector<int8_t> inZpVals     = { inZp };
    std::vector<int8_t> weightZpVals = { weightZp };
    std::vector<int32_t> biasVals    = { 0 };
    tb.setInput(inVals);
    tb.setInput(weightVals);
    tb.setInput(biasVals);
    tb.setInput(inZpVals);
    tb.setInput(weightZpVals);

    REQUIRE(tb.run() == GraphStatus::TOSA_VALID);
    std::vector<int32_t> actualOut = tb.getOutput<int32_t>(0, /* size */ expectedOut.size());

    compareOutput<int32_t>(expectedOut, actualOut);
}

void testTransposeConv2dBias(std::vector<int8_t>& inVals,
                             std::vector<int8_t>& weightVals,
                             std::vector<int32_t>& biasVals,
                             int8_t inZp,
                             int8_t weightZp,
                             std::vector<int32_t>& expectedOut)
{
    RefModelTestBuilder tb{};
    const DType inDtype  = DType_INT8;
    const DType outDtype = DType_INT32;

    const int IN_CHANNELS  = 2;
    const int OUT_CHANNELS = 2;

    tb.addInput({ 1, 1, 1, IN_CHANNELS }, inDtype);                      // input
    tb.addInput({ OUT_CHANNELS, 1, 1, IN_CHANNELS }, inDtype);           // weight
    tb.addInput({ static_cast<int32_t>(biasVals.size()) }, outDtype);    // bias
    tb.addInput({ 1 }, inDtype);                                         // input zp
    tb.addInput({ 1 }, inDtype);                                         // weight zp
    tb.addOutput({ 1, 1, 1, OUT_CHANNELS }, outDtype);

    auto attr = TosaTransposeConv2dAttribute(/* pad */ { 0, 0, 0, 0 },
                                             /* stride */ { 1, 1 },
                                             /* local_bound */ true,
                                             /* acc_type */ DType_INT32);

    tb.addOp(Op_TRANSPOSE_CONV2D, Attribute_TransposeConv2dAttribute, &attr);

    tb.initializeRunner();

    std::vector<int8_t> inZpVals     = { inZp };
    std::vector<int8_t> weightZpVals = { weightZp };

    tb.setInput(inVals);
    tb.setInput(weightVals);
    tb.setInput(biasVals);
    tb.setInput(inZpVals);
    tb.setInput(weightZpVals);

    REQUIRE(tb.run() == GraphStatus::TOSA_VALID);
    std::vector<int32_t> actualOut = tb.getOutput<int32_t>(0, /* size */ expectedOut.size());

    compareOutput<int32_t>(expectedOut, actualOut);
}

void testConv3d(std::vector<int8_t>& inVals, std::vector<int8_t>& weightVals, int8_t inZp, int8_t weightZp)
{
    RefModelTestBuilder tb{};
    constexpr DType inDtype     = DType_INT8;
    constexpr DType weightDtype = DType_INT8;
    constexpr DType outDtype    = DType_INT32;

    const int DEPTH  = 2;
    const int HEIGHT = 2;
    const int WIDTH  = 2;

    REQUIRE_MESSAGE(inVals.size() == DEPTH * HEIGHT * WIDTH,
                    "Unit test construction error: testConv3d assumes the inVals has ", DEPTH * HEIGHT * WIDTH,
                    " elements");
    REQUIRE_MESSAGE(weightVals.size() == DEPTH * HEIGHT * WIDTH,
                    "Unit test construction error: testConv3d assumes the weightVals has ", DEPTH * HEIGHT * WIDTH,
                    " elements");

    tb.addInput({ 1, DEPTH, HEIGHT, WIDTH, 1 }, inDtype);
    tb.addInput({ 1, DEPTH, HEIGHT, WIDTH, 1 }, weightDtype);
    tb.addInput({ 1 }, outDtype);    // bias
    tb.addInput({ 1 }, inDtype);
    tb.addInput({ 1 }, weightDtype);
    tb.addOutput({ 1, 1, 1, 1, 1 }, outDtype);

    auto attr = TosaConv3dAttribute({ 0, 0, 0, 0, 0, 0 }, { 1, 1, 1 }, { 1, 1, 1 }, true, DType_INT32);
    tb.addOp(Op_CONV3D, Attribute_Conv3dAttribute, &attr);

    tb.initializeRunner();

    int32_t expectedOutVal = 0;
    for (size_t i = 0; i < inVals.size(); i++)
    {
        expectedOutVal += (static_cast<int32_t>(inVals[i]) - static_cast<int32_t>(inZp)) *
                          (static_cast<int32_t>(weightVals[i]) - static_cast<int32_t>(weightZp));
    }
    std::vector<int32_t> expectedOut = { expectedOutVal };

    std::vector<int8_t> inZpVals     = { inZp };
    std::vector<int8_t> weightZpVals = { weightZp };
    std::vector<int32_t> biasVals    = { 0 };
    tb.setInput(inVals);
    tb.setInput(weightVals);
    tb.setInput(biasVals);
    tb.setInput(inZpVals);
    tb.setInput(weightZpVals);

    REQUIRE(tb.run() == GraphStatus::TOSA_VALID);
    std::vector<int32_t> actualOut = tb.getOutput<int32_t>(0, /* size */ expectedOut.size());

    compareOutput<int32_t>(expectedOut, actualOut);
}

void testMatMul(std::vector<int8_t>& aVals, std::vector<int8_t>& bVals, int8_t aZp, int8_t bZp)
{
    RefModelTestBuilder tb{};
    constexpr DType aDtype   = DType_INT8;
    constexpr DType bDtype   = DType_INT8;
    constexpr DType aZpDtype = DType_INT8;
    constexpr DType bZpDtype = DType_INT8;
    constexpr DType outDtype = DType_INT32;

    const int H = 2;
    const int C = 2;
    const int W = 2;

    REQUIRE_MESSAGE(aVals.size() == H * C, "Unit test construction error: testMatMulOverflow assumes the aVals has ",
                    H * C, " elements");
    REQUIRE_MESSAGE(bVals.size() == C * W, "Unit test construction error: testMatMulOverflow assumes the bVals has ",
                    C * W, " elements");

    tb.addInput({ 1, H, C }, aDtype);
    tb.addInput({ 1, C, W }, bDtype);
    tb.addInput({ 1 }, aZpDtype);
    tb.addInput({ 1 }, bZpDtype);
    tb.addOutput({ 1, H, W }, outDtype);

    auto attr = TosaMatMulAttribute();
    tb.addOp(Op_MATMUL, Attribute_MatMulAttribute, &attr);

    tb.initializeRunner();

    std::vector<int32_t> expectedOutVal(H * W, 0);

    for (int row = 0; row < H; ++row)
    {
        for (int col = 0; col < W; ++col)
        {
            int32_t dotProduct = 0;
            for (int k = 0; k < C; ++k)
            {
                int32_t aValue = static_cast<int32_t>(aVals[row * C + k]) - static_cast<int32_t>(aZp);
                int32_t bValue = static_cast<int32_t>(bVals[k * W + col]) - static_cast<int32_t>(bZp);
                dotProduct += aValue * bValue;
            }
            expectedOutVal[row * W + col] = dotProduct;
        }
    }
    std::vector<int32_t> expectedOut = { expectedOutVal };

    std::vector<int8_t> aZpVals = { aZp };
    std::vector<int8_t> bZpVals = { bZp };
    tb.setInput(aVals);
    tb.setInput(bVals);
    tb.setInput(aZpVals);
    tb.setInput(bZpVals);

    REQUIRE(tb.run() == GraphStatus::TOSA_VALID);
    std::vector<int32_t> actualOut = tb.getOutput<int32_t>(0, /* size */ expectedOut.size());

    compareOutput<int32_t>(expectedOut, actualOut);
}

void testAvgPool2d(std::vector<int8_t>& inVals, int8_t inZp, int8_t outZp)
{
    RefModelTestBuilder tb{};
    constexpr DType inDtype  = DType_INT8;
    constexpr DType outDtype = DType_INT8;

    // Parameters were chosen to have a single output value to simplify the computation
    const int IH            = 2;
    const int IW            = 2;
    const int KERNEL_HEIGHT = 2;
    const int KERNEL_WIDTH  = 2;
    const int STRIDE_HEIGHT = 2;
    const int STRIDE_WIDTH  = 2;
    const int PAD_TOP       = 0;
    const int PAD_BOTTOM    = 0;
    const int PAD_LEFT      = 0;
    const int PAD_RIGHT     = 0;
    // const int OH            = (IH + PAD_TOP + PAD_BOTTOM - KERNEL_HEIGHT) / STRIDE_HEIGHT + 1;
    // const int OW            = (IW + PAD_LEFT + PAD_RIGHT - KERNEL_WIDTH) / STRIDE_WIDTH + 1;
    const int OH = 1;
    const int OW = 1;

    REQUIRE_MESSAGE(inVals.size() == IH * IW,
                    "Unit test construction error: testAvgPool2dOverflow assumes the inVals has ", IH * IW,
                    " elements");

    tb.addInput({ 1, IH, IW, 1 }, inDtype);
    tb.addInput({ 1 }, inDtype);
    tb.addInput({ 1 }, outDtype);
    tb.addOutput({ 1, OH, OW, 1 }, outDtype);

    auto attr = TosaAvgPool2dAttribute({ KERNEL_HEIGHT, KERNEL_WIDTH }, { STRIDE_HEIGHT, STRIDE_WIDTH },
                                       { PAD_TOP, PAD_BOTTOM, PAD_LEFT, PAD_RIGHT }, DType_INT32);
    tb.addOp(Op_AVG_POOL2D, Attribute_AvgPool2dAttribute, &attr);

    tb.initializeRunner();
    int8_t expectedOutVal = 0;

    int32_t sum = 0;
    for (const auto v : inVals)
    {
        sum += static_cast<int32_t>(v) - static_cast<int32_t>(inZp);
    }

    // NOTE: The following calculations are optimized for a count of 4 elements
    // (KERNEL_HEIGHT * KERNEL_WIDTH = 4) in the computation
    // which leads to a simplified implementation of the spec
    // scale_t scale = reciprocal_scale(count);  (spec)
    int8_t shift       = 32;
    int32_t multiplier = (1 << 30) + 1;
    // apply_scale_32(sum, multiplier, shift, /* double_round */ false); (spec)
    int64_t round  = 1LL << (shift - 1);
    int64_t result = (static_cast<int64_t>(sum) * multiplier) + round;
    result >>= shift;
    int32_t final_result = static_cast<int32_t>(result);
    // apply_clip_s<acc_t>(acc, minimum_s<in_out_t>(), maximum_s<in_out_t>()); (spec)
    expectedOutVal = static_cast<int8_t>(std::clamp(final_result + outZp, -128, 127));

    std::vector<int8_t> expectedOut = { expectedOutVal };
    tb.setInput(inVals);
    std::vector<int8_t> inZpVals  = { inZp };
    std::vector<int8_t> outZpVals = { outZp };
    tb.setInput(inZpVals);
    tb.setInput(outZpVals);

    REQUIRE(tb.run() == GraphStatus::TOSA_VALID);
    std::vector<int8_t> actualOut = tb.getOutput<int8_t>(0, /* size */ expectedOut.size());

    compareOutput<int8_t>(expectedOut, actualOut);
}

template <typename FP_TYPE>
void testPow(std::vector<FP_TYPE>& inVals1, std::vector<FP_TYPE>& inVals2, std::vector<FP_TYPE>& expectedOut)
{
    RefModelTestBuilder tb{};
    constexpr DType dtype = NativeType2DType<FP_TYPE>();

    const bool sizeMatch = (inVals1.size() == inVals2.size()) && (inVals1.size() == expectedOut.size());
    REQUIRE_MESSAGE(sizeMatch, "test construction error: size mismatch input1: %d, input2: %d, and output: %d",
                    inVals1.size(), inVals2.size(), expectedOut.size());

    int32_t sz = static_cast<int32_t>(inVals1.size());
    tb.addInput({ sz }, dtype);
    tb.addInput({ sz }, dtype);
    tb.addOutput({ sz }, dtype);

    TosaPowAttribute attr{};
    tb.addOp(Op_POW, Attribute_PowAttribute, &attr);

    tb.initializeRunner();

    tb.setInput(inVals1);
    tb.setInput(inVals2);

    REQUIRE(tb.run() == GraphStatus::TOSA_VALID);
    std::vector<FP_TYPE> actualOut = tb.getOutput<FP_TYPE>(0, /* size */ expectedOut.size());

    compareOutput<FP_TYPE>(expectedOut, actualOut);
}

TEST_SUITE("reference_model")
{
    TEST_CASE_TEMPLATE("ARGMAX FP SPECIAL", IN_TYPE, float, half, bfloat16, fp8_e5m2, fp8_e4m3)
    {
        SUBCASE("propagate")
        {
            testArgmaxFpSpecial<IN_TYPE>(true);
        }
        SUBCASE("ignore")
        {
            testArgmaxFpSpecial<IN_TYPE>(false);
        }
    }

    TEST_CASE("CONV2D, DEPTHWISE_CONV2D, TRANSPOSE_CONV2D and MATMUL zero point avoids overflow")
    {
        SUBCASE("input negative overflow")
        {
            INFO("This test is meant to catch cases where the input zero point is subtracted from the input value "
                 "using an int8_t accumulator instead of a full-precision int32_t one");
            std::vector<int8_t> inVals     = { 126, -126, -120, 0 };
            std::vector<int8_t> weightVals = { 2, 1, 4, -7 };
            const int8_t inZp              = 4;
            const int8_t weightZp          = 15;

            testConv2d(inVals, weightVals, inZp, weightZp);
            testDepthwiseConv2d(inVals, weightVals, inZp, weightZp);
            testTransposeConv2d(inVals, weightVals, inZp, weightZp);
        }

        SUBCASE("input positive overflow")
        {
            INFO("This test is meant to catch cases where the input zero point is subtracted from the input value "
                 "using an int8_t accumulator instead of a full-precision int32_t one");
            std::vector<int8_t> inVals     = { 122, -7, -120, 121 };
            std::vector<int8_t> weightVals = { 2, 1, 4, -7 };
            const int8_t inZp              = -10;
            const int8_t weightZp          = -5;
            testConv2d(inVals, weightVals, inZp, weightZp);
            testDepthwiseConv2d(inVals, weightVals, inZp, weightZp);
            testTransposeConv2d(inVals, weightVals, inZp, weightZp);
        }

        SUBCASE("weight negative overflow")
        {
            INFO("This test is meant to catch cases where the weight zero point is subtracted from the weight value "
                 "using an int8_t accumulator instead of a full-precision int32_t one");
            std::vector<int8_t> inVals     = { 3, -12, 5, 70 };
            std::vector<int8_t> weightVals = { -120, -12, 0, -7 };
            const int8_t inZp              = 55;
            const int8_t weightZp          = 15;

            testConv2d(inVals, weightVals, inZp, weightZp);
            testDepthwiseConv2d(inVals, weightVals, inZp, weightZp);
            testTransposeConv2d(inVals, weightVals, inZp, weightZp);
        }

        SUBCASE("weight positive overflow")
        {
            INFO("This test is meant to catch cases where the weight zero point is subtracted from the weight value "
                 "using an int8_t accumulator instead of a full-precision int32_t one");
            std::vector<int8_t> inVals     = { -5, 65, -1, 32 };
            std::vector<int8_t> weightVals = { -2, 125, 4, -1 };
            const int8_t inZp              = -10;
            const int8_t weightZp          = -5;
            testConv2d(inVals, weightVals, inZp, weightZp);
            testDepthwiseConv2d(inVals, weightVals, inZp, weightZp);
            testTransposeConv2d(inVals, weightVals, inZp, weightZp);
        }
    }

    TEST_CASE("CONV3D zero point avoids overflow")
    {
        SUBCASE("input negative overflow")
        {
            INFO("This test is meant to catch cases where the input zero point is subtracted from the input value "
                 "using an int8_t accumulator instead of a full-precision int32_t one");
            std::vector<int8_t> inVals     = { 127, -128, 126, -127, 125, -126, 124, -125 };
            std::vector<int8_t> weightVals = { 2, -2, 1, -1, 3, -3, 4, -4 };
            const int8_t inZp              = 4;
            const int8_t weightZp          = 15;

            testConv3d(inVals, weightVals, inZp, weightZp);
        }

        SUBCASE("input positive overflow")
        {
            INFO("This test is meant to catch cases where the input zero point is subtracted from the input value "
                 "using an int8_t accumulator instead of a full-precision int32_t one");
            std::vector<int8_t> inVals     = { 122, -7, -120, 121, 123, -8, -119, 120 };
            std::vector<int8_t> weightVals = { 2, 1, 4, -7, 3, -2, 5, -6 };
            const int8_t inZp              = -10;
            const int8_t weightZp          = -5;
            testConv3d(inVals, weightVals, inZp, weightZp);
        }

        SUBCASE("weight negative overflow")
        {
            INFO("This test is meant to catch cases where the weight zero point is subtracted from the weight value "
                 "using an int8_t accumulator instead of a full-precision int32_t one");
            std::vector<int8_t> inVals     = { 3, -12, 5, 70, -10, 20, 15, 25 };
            std::vector<int8_t> weightVals = { -120, -12, 0, -7, -50, 10, -30, 5 };
            const int8_t inZp              = 55;
            const int8_t weightZp          = 15;

            testConv3d(inVals, weightVals, inZp, weightZp);
        }

        SUBCASE("weight positive overflow")
        {
            INFO("This test is meant to catch cases where the weight zero point is subtracted from the weight value "
                 "using an int8_t accumulator instead of a full-precision int32_t one");
            std::vector<int8_t> inVals     = { -5, 65, -1, 32, 20, -10, -15, 5 };
            std::vector<int8_t> weightVals = { -2, 125, 4, -1, -20, 30, 60, 120 };
            const int8_t inZp              = -10;
            const int8_t weightZp          = -5;
            testConv3d(inVals, weightVals, inZp, weightZp);
        }
    }

    TEST_CASE("CONV2D and TRANSPOSE_CONV2D broadcast bias correctly")
    {
        SUBCASE("broadcasted")
        {
            std::vector<int8_t> inVals     = { 5, 9 };
            std::vector<int8_t> weightVals = { 3, -7, -1, 4 };
            std::vector<int32_t> biasVals  = { -5 };

            std::vector<int32_t> expectedOut = {
                5 * 3 + 9 * (-7) + (-5),    // i1 * w11 + i2 * w12 + b
                5 * (-1) + 9 * 4 + (-5),    // i1 * w21 + i2 * w22 + b
            };

            testTransposeConv2dBias(inVals, weightVals, biasVals, /*in zp */ 0, /*weight zp*/ 0, expectedOut);
            testConv2dBias(inVals, weightVals, biasVals, /*in zp */ 0, /*weight zp*/ 0, expectedOut);
        }

        SUBCASE("not broadcasted")
        {
            std::vector<int8_t> inVals     = { 5, 9 };
            std::vector<int8_t> weightVals = { 3, -7, -1, 4 };
            std::vector<int32_t> biasVals  = { -5, 14 };

            std::vector<int32_t> expectedOut = {
                5 * 3 + 9 * (-7) + (-5),    // i1 * w11 + i2 * w12 + b1
                5 * (-1) + 9 * 4 + 14,      // i1 * w21 + i2 * w22 + b2
            };

            testTransposeConv2dBias(inVals, weightVals, biasVals, /*in zp */ 0, /*weight zp*/ 0, expectedOut);
            testConv2dBias(inVals, weightVals, biasVals, /*in zp */ 0, /*weight zp*/ 0, expectedOut);
        }
    }

    TEST_CASE("DEPTHWISE_CONV2D broadcasts bias correctly")
    {
        SUBCASE("broadcasted")
        {
            std::vector<int8_t> inVals     = { 5, 9 };
            std::vector<int8_t> weightVals = { 3, -7 };
            std::vector<int32_t> biasVals  = { -5 };

            std::vector<int32_t> expectedOut = {
                5 * 3 + (-5),       // i1 * w1 + b
                9 * (-7) + (-5),    // i2 * w2 + b
            };

            testDepthwiseConv2dBias(inVals, weightVals, biasVals, /*in zp */ 0, /*weight zp*/ 0, expectedOut);
        }

        SUBCASE("not broadcasted")
        {
            std::vector<int8_t> inVals     = { 5, 9 };
            std::vector<int8_t> weightVals = { 3, -7 };
            std::vector<int32_t> biasVals  = { -5, 14 };

            std::vector<int32_t> expectedOut = {
                5 * 3 + (-5),     // i1 * w1 + b1
                9 * (-7) + 14,    // i2 * w2 + b2
            };

            testDepthwiseConv2dBias(inVals, weightVals, biasVals, /*in zp */ 0, /*weight zp*/ 0, expectedOut);
        }
    }

    TEST_CASE("AVG_POOL2D zero point avoids overflow")
    {
        SUBCASE("input negative overflow")
        {
            INFO("This test is meant to catch cases where the input zero point is subtracted from the input value "
                 "using an int8_t accumulator instead of a full-precision int32_t one");
            std::vector<int8_t> inVals = { 126, -126, -128, 0 };
            const int8_t inZp          = 4;
            const int8_t outZp         = 15;

            testAvgPool2d(inVals, inZp, outZp);
        }

        SUBCASE("input positive overflow")
        {
            INFO("This test is meant to catch cases where the input zero point is subtracted from the input value "
                 "using an int8_t accumulator instead of a full-precision int32_t one");
            std::vector<int8_t> inVals = { 126, 126, 127, 127 };
            const int8_t inZp          = -12;
            const int8_t outZp         = -5;

            testAvgPool2d(inVals, inZp, outZp);
        }
    }
}    // TEST_SUITE("reference_model")


// Copyright (c) 2022, ARM Limited.
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

// THIS FILE IS GENERATED. DO NOT EDIT!
// See scripts/operator_api/generate_api.py

#include "operators.h"
#include "model_runner_impl.h"
#include "ops/op_factory.h"

#define TOSA_RETURN_ON_ERROR(status)                                                                                   \
    do                                                                                                                 \
    {                                                                                                                  \
        if (status != 0)                                                                                               \
        {                                                                                                              \
            return tosa_status_error;                                                                                  \
        }                                                                                                              \
    } while (false)

#define TOSA_RETURN_ON_GRAPH_STATUS_ERROR(status)                                                                      \
    do                                                                                                                 \
    {                                                                                                                  \
        if (status != GraphStatus::TOSA_VALID)                                                                         \
        {                                                                                                              \
            auto ustatus = static_cast<std::underlying_type_t<GraphStatus>>(status);                                   \
            return static_cast<tosa_status_t>(ustatus);                                                                \
        }                                                                                                              \
    } while (false)

namespace
{

tosa::DType translate_client_datatype(tosa_datatype_t type)
{
    switch (type)
    {
        case tosa_datatype_fp16_t:
            return tosa::DType::DType_FP16;
        case tosa_datatype_fp32_t:
            return tosa::DType::DType_FP32;
        default:
            return tosa::DType::DType_UNKNOWN;
    }
};

tosa::TosaSerializationTensor* translate_client_tensor(tosa_tensor_t& tensor, const std::string& name)
{
    std::vector<int32_t> shape(tensor.shape, tensor.shape + tensor.num_dims);
    return new tosa::TosaSerializationTensor(name, shape, translate_client_datatype(tensor.data_type), {});
}

tosa::ResizeMode translate_client_tosa_mode(tosa_mode_t mode)
{
    switch (mode)
    {
        case tosa_mode_nearest:
            return tosa::ResizeMode_NEAREST;
        case tosa_mode_max:
        case tosa_mode_bilinear:
            return tosa::ResizeMode_BILINEAR;
        default:
            return tosa::ResizeMode_UNKNOWN;
    }
}

}    // namespace

extern "C"
{

    tosa_status_t tosa_run_argmax(tosa_tensor_t client_input, const int32_t client_axis, tosa_tensor_t client_output)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        tosa::TosaSerializationTensor* input  = translate_client_tensor(client_input, "input");
        tosa::TosaSerializationTensor* output = translate_client_tensor(client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_ARGMAX, tosa::Attribute::Attribute_NONE, &attr,
                                                      { input->GetName() }, { output->GetName() });

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("argmax", { op }, { input, output }, { input->GetName() },
                                                { output->GetName() });

        // Setup model
        TosaReference::ModelRunnerImpl runner;
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));
        TOSA_RETURN_ON_ERROR(runner.setInput(input->GetName(), client_input.data, client_input.size));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(runner.getOutput(output->GetName(), client_output.data, client_output.size));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_avg_pool2d(tosa_tensor_t client_input,
                                      const int32_t client_kernel[2],
                                      const int32_t client_stride[2],
                                      const int32_t client_pad[4],
                                      const int32_t client_input_zp,
                                      const int32_t client_output_zp,
                                      tosa_tensor_t client_output)
    {
        // Create operator attributes
        const std::vector<int32_t> pad(&client_pad[0], &client_pad[4]);
        const std::vector<int32_t> kernel(&client_kernel[0], &client_kernel[2]);
        const std::vector<int32_t> stride(&client_stride[0], &client_stride[2]);
        const int32_t input_zp        = client_input_zp;
        const int32_t output_zp       = client_output_zp;
        const tosa::DType accum_dtype = tosa::DType::DType_FP32;
        TosaPoolAttribute attr(pad, kernel, stride, input_zp, output_zp, accum_dtype);

        // Create tensors
        tosa::TosaSerializationTensor* input  = translate_client_tensor(client_input, "input");
        tosa::TosaSerializationTensor* output = translate_client_tensor(client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_AVG_POOL2D, tosa::Attribute::Attribute_PoolAttribute,
                                                      &attr, { input->GetName() }, { output->GetName() });

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("avg_pool2d", { op }, { input, output }, { input->GetName() },
                                                { output->GetName() });

        // Setup model
        TosaReference::ModelRunnerImpl runner;
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));
        TOSA_RETURN_ON_ERROR(runner.setInput(input->GetName(), client_input.data, client_input.size));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(runner.getOutput(output->GetName(), client_output.data, client_output.size));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_conv2d(tosa_tensor_t client_input,
                                  tosa_tensor_t client_weight,
                                  tosa_tensor_t client_bias,
                                  const int32_t client_pad[4],
                                  const int32_t client_stride[2],
                                  const int32_t client_dilation[2],
                                  const int32_t client_input_zp,
                                  const int32_t client_weight_zp,
                                  tosa_tensor_t client_output)
    {
        // Create operator attributes
        const std::vector<int32_t> pad(&client_pad[0], &client_pad[4]);
        const std::vector<int32_t> stride(&client_stride[0], &client_stride[2]);
        const std::vector<int32_t> dilation(&client_dilation[0], &client_dilation[2]);
        const int32_t input_zp        = client_input_zp;
        const int32_t weight_zp       = client_weight_zp;
        const tosa::DType accum_dtype = tosa::DType::DType_FP32;
        TosaConvAttribute attr(pad, stride, dilation, input_zp, weight_zp, accum_dtype);

        // Create tensors
        tosa::TosaSerializationTensor* input  = translate_client_tensor(client_input, "input");
        tosa::TosaSerializationTensor* weight = translate_client_tensor(client_weight, "weight");
        tosa::TosaSerializationTensor* bias   = translate_client_tensor(client_bias, "bias");
        tosa::TosaSerializationTensor* output = translate_client_tensor(client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_CONV2D, tosa::Attribute::Attribute_ConvAttribute,
                                                      &attr, { input->GetName(), weight->GetName(), bias->GetName() },
                                                      { output->GetName() });

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("conv2d", { op }, { input, weight, bias, output },
                                                { input->GetName(), weight->GetName(), bias->GetName() },
                                                { output->GetName() });

        // Setup model
        TosaReference::ModelRunnerImpl runner;
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));
        TOSA_RETURN_ON_ERROR(runner.setInput(input->GetName(), client_input.data, client_input.size));
        TOSA_RETURN_ON_ERROR(runner.setInput(weight->GetName(), client_weight.data, client_weight.size));
        TOSA_RETURN_ON_ERROR(runner.setInput(bias->GetName(), client_bias.data, client_bias.size));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(runner.getOutput(output->GetName(), client_output.data, client_output.size));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_conv3d(tosa_tensor_t client_input,
                                  tosa_tensor_t client_weight,
                                  tosa_tensor_t client_bias,
                                  const int32_t client_pad[6],
                                  const int32_t client_stride[3],
                                  const int32_t client_dilation[3],
                                  const int32_t client_input_zp,
                                  const int32_t client_weight_zp,
                                  tosa_tensor_t client_output)
    {
        // Create operator attributes
        const std::vector<int32_t> pad(&client_pad[0], &client_pad[6]);
        const std::vector<int32_t> stride(&client_stride[0], &client_stride[3]);
        const std::vector<int32_t> dilation(&client_dilation[0], &client_dilation[3]);
        const int32_t input_zp        = client_input_zp;
        const int32_t weight_zp       = client_weight_zp;
        const tosa::DType accum_dtype = tosa::DType::DType_FP32;
        TosaConvAttribute attr(pad, stride, dilation, input_zp, weight_zp, accum_dtype);

        // Create tensors
        tosa::TosaSerializationTensor* input  = translate_client_tensor(client_input, "input");
        tosa::TosaSerializationTensor* weight = translate_client_tensor(client_weight, "weight");
        tosa::TosaSerializationTensor* bias   = translate_client_tensor(client_bias, "bias");
        tosa::TosaSerializationTensor* output = translate_client_tensor(client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_CONV3D, tosa::Attribute::Attribute_ConvAttribute,
                                                      &attr, { input->GetName(), weight->GetName(), bias->GetName() },
                                                      { output->GetName() });

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("conv3d", { op }, { input, weight, bias, output },
                                                { input->GetName(), weight->GetName(), bias->GetName() },
                                                { output->GetName() });

        // Setup model
        TosaReference::ModelRunnerImpl runner;
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));
        TOSA_RETURN_ON_ERROR(runner.setInput(input->GetName(), client_input.data, client_input.size));
        TOSA_RETURN_ON_ERROR(runner.setInput(weight->GetName(), client_weight.data, client_weight.size));
        TOSA_RETURN_ON_ERROR(runner.setInput(bias->GetName(), client_bias.data, client_bias.size));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(runner.getOutput(output->GetName(), client_output.data, client_output.size));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_depthwise_conv2d(tosa_tensor_t client_input,
                                            tosa_tensor_t client_weight,
                                            tosa_tensor_t client_bias,
                                            const int32_t client_pad[4],
                                            const int32_t client_stride[2],
                                            const int32_t client_dilation[2],
                                            const int32_t client_input_zp,
                                            const int32_t client_weight_zp,
                                            tosa_tensor_t client_output)
    {
        // Create operator attributes
        const std::vector<int32_t> pad(&client_pad[0], &client_pad[4]);
        const std::vector<int32_t> stride(&client_stride[0], &client_stride[2]);
        const std::vector<int32_t> dilation(&client_dilation[0], &client_dilation[2]);
        const int32_t input_zp        = client_input_zp;
        const int32_t weight_zp       = client_weight_zp;
        const tosa::DType accum_dtype = tosa::DType::DType_FP32;
        TosaConvAttribute attr(pad, stride, dilation, input_zp, weight_zp, accum_dtype);

        // Create tensors
        tosa::TosaSerializationTensor* input  = translate_client_tensor(client_input, "input");
        tosa::TosaSerializationTensor* weight = translate_client_tensor(client_weight, "weight");
        tosa::TosaSerializationTensor* bias   = translate_client_tensor(client_bias, "bias");
        tosa::TosaSerializationTensor* output = translate_client_tensor(client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(
            tosa::Op::Op_DEPTHWISE_CONV2D, tosa::Attribute::Attribute_ConvAttribute, &attr,
            { input->GetName(), weight->GetName(), bias->GetName() }, { output->GetName() });

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("depthwise_conv2d", { op }, { input, weight, bias, output },
                                                { input->GetName(), weight->GetName(), bias->GetName() },
                                                { output->GetName() });

        // Setup model
        TosaReference::ModelRunnerImpl runner;
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));
        TOSA_RETURN_ON_ERROR(runner.setInput(input->GetName(), client_input.data, client_input.size));
        TOSA_RETURN_ON_ERROR(runner.setInput(weight->GetName(), client_weight.data, client_weight.size));
        TOSA_RETURN_ON_ERROR(runner.setInput(bias->GetName(), client_bias.data, client_bias.size));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(runner.getOutput(output->GetName(), client_output.data, client_output.size));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_fully_connected(tosa_tensor_t client_input,
                                           const int32_t client_input_zp,
                                           const int32_t client_weight_zp,
                                           tosa_tensor_t client_output)
    {
        // Create operator attributes
        const int32_t input_zp        = client_input_zp;
        const int32_t weight_zp       = client_weight_zp;
        const tosa::DType accum_dtype = tosa::DType::DType_FP32;
        TosaFullyConnectedAttribute attr(input_zp, weight_zp, accum_dtype);

        // Create tensors
        tosa::TosaSerializationTensor* input  = translate_client_tensor(client_input, "input");
        tosa::TosaSerializationTensor* output = translate_client_tensor(client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_FULLY_CONNECTED,
                                                      tosa::Attribute::Attribute_FullyConnectedAttribute, &attr,
                                                      { input->GetName() }, { output->GetName() });

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("fully_connected", { op }, { input, output }, { input->GetName() },
                                                { output->GetName() });

        // Setup model
        TosaReference::ModelRunnerImpl runner;
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));
        TOSA_RETURN_ON_ERROR(runner.setInput(input->GetName(), client_input.data, client_input.size));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(runner.getOutput(output->GetName(), client_output.data, client_output.size));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_matmul(tosa_tensor_t client_a,
                                  tosa_tensor_t client_b,
                                  const int32_t client_a_zp,
                                  const int32_t client_b_zp,
                                  tosa_tensor_t client_output)
    {
        // Create operator attributes
        const int32_t a_zp            = client_a_zp;
        const int32_t b_zp            = client_b_zp;
        const tosa::DType accum_dtype = tosa::DType::DType_FP32;
        TosaMatMulAttribute attr(a_zp, b_zp, accum_dtype);

        // Create tensors
        tosa::TosaSerializationTensor* a      = translate_client_tensor(client_a, "a");
        tosa::TosaSerializationTensor* b      = translate_client_tensor(client_b, "b");
        tosa::TosaSerializationTensor* output = translate_client_tensor(client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_MATMUL, tosa::Attribute::Attribute_MatMulAttribute,
                                                      &attr, { a->GetName(), b->GetName() }, { output->GetName() });

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("matmul", { op }, { a, b, output }, { a->GetName(), b->GetName() },
                                                { output->GetName() });

        // Setup model
        TosaReference::ModelRunnerImpl runner;
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));
        TOSA_RETURN_ON_ERROR(runner.setInput(a->GetName(), client_a.data, client_a.size));
        TOSA_RETURN_ON_ERROR(runner.setInput(b->GetName(), client_b.data, client_b.size));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(runner.getOutput(output->GetName(), client_output.data, client_output.size));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_max_pool2d(tosa_tensor_t client_input,
                                      const int32_t client_kernel[2],
                                      const int32_t client_stride[2],
                                      const int32_t client_pad[4],
                                      const int32_t client_input_zp,
                                      const int32_t client_output_zp,
                                      tosa_tensor_t client_output)
    {
        // Create operator attributes
        const std::vector<int32_t> pad(&client_pad[0], &client_pad[4]);
        const std::vector<int32_t> kernel(&client_kernel[0], &client_kernel[2]);
        const std::vector<int32_t> stride(&client_stride[0], &client_stride[2]);
        const int32_t input_zp        = client_input_zp;
        const int32_t output_zp       = client_output_zp;
        const tosa::DType accum_dtype = tosa::DType::DType_FP32;
        TosaPoolAttribute attr(pad, kernel, stride, input_zp, output_zp, accum_dtype);

        // Create tensors
        tosa::TosaSerializationTensor* input  = translate_client_tensor(client_input, "input");
        tosa::TosaSerializationTensor* output = translate_client_tensor(client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_MAX_POOL2D, tosa::Attribute::Attribute_PoolAttribute,
                                                      &attr, { input->GetName() }, { output->GetName() });

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("max_pool2d", { op }, { input, output }, { input->GetName() },
                                                { output->GetName() });

        // Setup model
        TosaReference::ModelRunnerImpl runner;
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));
        TOSA_RETURN_ON_ERROR(runner.setInput(input->GetName(), client_input.data, client_input.size));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(runner.getOutput(output->GetName(), client_output.data, client_output.size));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_transpose_conv2d(tosa_tensor_t client_input,
                                            tosa_tensor_t client_weight,
                                            tosa_tensor_t client_bias,
                                            const int32_t client_out_pad[4],
                                            const int32_t client_stride[2],
                                            const int32_t client_out_shape[4],
                                            const int32_t client_input_zp,
                                            const int32_t client_weight_zp,
                                            const int32_t client_pad_len,
                                            const int32_t client_pad[],
                                            const int32_t client_dilation_len,
                                            const int32_t client_dilation[],
                                            tosa_tensor_t client_output)
    {
        // Create operator attributes
        const std::vector<int32_t> pad(&client_pad[0], &client_pad[0] + client_pad_len);
        const std::vector<int32_t> stride(&client_stride[0], &client_stride[2]);
        const std::vector<int32_t> dilation(&client_dilation[0], &client_dilation[0] + client_dilation_len);
        const int32_t input_zp        = client_input_zp;
        const int32_t weight_zp       = client_weight_zp;
        const tosa::DType accum_dtype = tosa::DType::DType_FP32;
        TosaConvAttribute attr(pad, stride, dilation, input_zp, weight_zp, accum_dtype);

        // Create tensors
        tosa::TosaSerializationTensor* input  = translate_client_tensor(client_input, "input");
        tosa::TosaSerializationTensor* weight = translate_client_tensor(client_weight, "weight");
        tosa::TosaSerializationTensor* bias   = translate_client_tensor(client_bias, "bias");
        tosa::TosaSerializationTensor* output = translate_client_tensor(client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(
            tosa::Op::Op_TRANSPOSE_CONV2D, tosa::Attribute::Attribute_ConvAttribute, &attr,
            { input->GetName(), weight->GetName(), bias->GetName() }, { output->GetName() });

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("transpose_conv2d", { op }, { input, weight, bias, output },
                                                { input->GetName(), weight->GetName(), bias->GetName() },
                                                { output->GetName() });

        // Setup model
        TosaReference::ModelRunnerImpl runner;
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));
        TOSA_RETURN_ON_ERROR(runner.setInput(input->GetName(), client_input.data, client_input.size));
        TOSA_RETURN_ON_ERROR(runner.setInput(weight->GetName(), client_weight.data, client_weight.size));
        TOSA_RETURN_ON_ERROR(runner.setInput(bias->GetName(), client_bias.data, client_bias.size));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(runner.getOutput(output->GetName(), client_output.data, client_output.size));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_clamp(tosa_tensor_t client_input,
                                 const int32_t client_min_int,
                                 const int32_t client_max_int,
                                 const float client_min_fp,
                                 const float client_max_fp,
                                 tosa_tensor_t client_output)
    {
        // Create operator attributes
        const int32_t min_int = client_min_int;
        const int32_t max_int = client_max_int;
        const float min_fp    = client_min_fp;
        const float max_fp    = client_max_fp;
        TosaClampAttribute attr(min_int, max_int, min_fp, max_fp);

        // Create tensors
        tosa::TosaSerializationTensor* input  = translate_client_tensor(client_input, "input");
        tosa::TosaSerializationTensor* output = translate_client_tensor(client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_CLAMP, tosa::Attribute::Attribute_ClampAttribute,
                                                      &attr, { input->GetName() }, { output->GetName() });

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("clamp", { op }, { input, output }, { input->GetName() },
                                                { output->GetName() });

        // Setup model
        TosaReference::ModelRunnerImpl runner;
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));
        TOSA_RETURN_ON_ERROR(runner.setInput(input->GetName(), client_input.data, client_input.size));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(runner.getOutput(output->GetName(), client_output.data, client_output.size));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_sigmoid(tosa_tensor_t client_input, tosa_tensor_t client_output)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        tosa::TosaSerializationTensor* input  = translate_client_tensor(client_input, "input");
        tosa::TosaSerializationTensor* output = translate_client_tensor(client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_SIGMOID, tosa::Attribute::Attribute_NONE, &attr,
                                                      { input->GetName() }, { output->GetName() });

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("sigmoid", { op }, { input, output }, { input->GetName() },
                                                { output->GetName() });

        // Setup model
        TosaReference::ModelRunnerImpl runner;
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));
        TOSA_RETURN_ON_ERROR(runner.setInput(input->GetName(), client_input.data, client_input.size));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(runner.getOutput(output->GetName(), client_output.data, client_output.size));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_tanh(tosa_tensor_t client_input, tosa_tensor_t client_output)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        tosa::TosaSerializationTensor* input  = translate_client_tensor(client_input, "input");
        tosa::TosaSerializationTensor* output = translate_client_tensor(client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_TANH, tosa::Attribute::Attribute_NONE, &attr,
                                                      { input->GetName() }, { output->GetName() });

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("tanh", { op }, { input, output }, { input->GetName() },
                                                { output->GetName() });

        // Setup model
        TosaReference::ModelRunnerImpl runner;
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));
        TOSA_RETURN_ON_ERROR(runner.setInput(input->GetName(), client_input.data, client_input.size));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(runner.getOutput(output->GetName(), client_output.data, client_output.size));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_add(tosa_tensor_t client_input1, tosa_tensor_t client_input2, tosa_tensor_t client_output)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        tosa::TosaSerializationTensor* input1 = translate_client_tensor(client_input1, "input1");
        tosa::TosaSerializationTensor* input2 = translate_client_tensor(client_input2, "input2");
        tosa::TosaSerializationTensor* output = translate_client_tensor(client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_ADD, tosa::Attribute::Attribute_NONE, &attr,
                                                      { input1->GetName(), input2->GetName() }, { output->GetName() });

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("add", { op }, { input1, input2, output },
                                                { input1->GetName(), input2->GetName() }, { output->GetName() });

        // Setup model
        TosaReference::ModelRunnerImpl runner;
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));
        TOSA_RETURN_ON_ERROR(runner.setInput(input1->GetName(), client_input1.data, client_input1.size));
        TOSA_RETURN_ON_ERROR(runner.setInput(input2->GetName(), client_input2.data, client_input2.size));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(runner.getOutput(output->GetName(), client_output.data, client_output.size));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_arithmetic_right_shift(tosa_tensor_t client_input1,
                                                  tosa_tensor_t client_input2,
                                                  const bool client_round,
                                                  tosa_tensor_t client_output)
    {
        // Create operator attributes
        const bool round = client_round;
        TosaArithmeticRightShiftAttribute attr(round);

        // Create tensors
        tosa::TosaSerializationTensor* input1 = translate_client_tensor(client_input1, "input1");
        tosa::TosaSerializationTensor* input2 = translate_client_tensor(client_input2, "input2");
        tosa::TosaSerializationTensor* output = translate_client_tensor(client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_ARITHMETIC_RIGHT_SHIFT,
                                                      tosa::Attribute::Attribute_ArithmeticRightShiftAttribute, &attr,
                                                      { input1->GetName(), input2->GetName() }, { output->GetName() });

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("arithmetic_right_shift", { op }, { input1, input2, output },
                                                { input1->GetName(), input2->GetName() }, { output->GetName() });

        // Setup model
        TosaReference::ModelRunnerImpl runner;
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));
        TOSA_RETURN_ON_ERROR(runner.setInput(input1->GetName(), client_input1.data, client_input1.size));
        TOSA_RETURN_ON_ERROR(runner.setInput(input2->GetName(), client_input2.data, client_input2.size));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(runner.getOutput(output->GetName(), client_output.data, client_output.size));

        return tosa_status_valid;
    }

    tosa_status_t
        tosa_run_bitwise_and(tosa_tensor_t client_input1, tosa_tensor_t client_input2, tosa_tensor_t client_output)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        tosa::TosaSerializationTensor* input1 = translate_client_tensor(client_input1, "input1");
        tosa::TosaSerializationTensor* input2 = translate_client_tensor(client_input2, "input2");
        tosa::TosaSerializationTensor* output = translate_client_tensor(client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_BITWISE_AND, tosa::Attribute::Attribute_NONE, &attr,
                                                      { input1->GetName(), input2->GetName() }, { output->GetName() });

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("bitwise_and", { op }, { input1, input2, output },
                                                { input1->GetName(), input2->GetName() }, { output->GetName() });

        // Setup model
        TosaReference::ModelRunnerImpl runner;
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));
        TOSA_RETURN_ON_ERROR(runner.setInput(input1->GetName(), client_input1.data, client_input1.size));
        TOSA_RETURN_ON_ERROR(runner.setInput(input2->GetName(), client_input2.data, client_input2.size));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(runner.getOutput(output->GetName(), client_output.data, client_output.size));

        return tosa_status_valid;
    }

    tosa_status_t
        tosa_run_bitwise_or(tosa_tensor_t client_input1, tosa_tensor_t client_input2, tosa_tensor_t client_output)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        tosa::TosaSerializationTensor* input1 = translate_client_tensor(client_input1, "input1");
        tosa::TosaSerializationTensor* input2 = translate_client_tensor(client_input2, "input2");
        tosa::TosaSerializationTensor* output = translate_client_tensor(client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_BITWISE_OR, tosa::Attribute::Attribute_NONE, &attr,
                                                      { input1->GetName(), input2->GetName() }, { output->GetName() });

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("bitwise_or", { op }, { input1, input2, output },
                                                { input1->GetName(), input2->GetName() }, { output->GetName() });

        // Setup model
        TosaReference::ModelRunnerImpl runner;
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));
        TOSA_RETURN_ON_ERROR(runner.setInput(input1->GetName(), client_input1.data, client_input1.size));
        TOSA_RETURN_ON_ERROR(runner.setInput(input2->GetName(), client_input2.data, client_input2.size));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(runner.getOutput(output->GetName(), client_output.data, client_output.size));

        return tosa_status_valid;
    }

    tosa_status_t
        tosa_run_bitwise_xor(tosa_tensor_t client_input1, tosa_tensor_t client_input2, tosa_tensor_t client_output)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        tosa::TosaSerializationTensor* input1 = translate_client_tensor(client_input1, "input1");
        tosa::TosaSerializationTensor* input2 = translate_client_tensor(client_input2, "input2");
        tosa::TosaSerializationTensor* output = translate_client_tensor(client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_BITWISE_XOR, tosa::Attribute::Attribute_NONE, &attr,
                                                      { input1->GetName(), input2->GetName() }, { output->GetName() });

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("bitwise_xor", { op }, { input1, input2, output },
                                                { input1->GetName(), input2->GetName() }, { output->GetName() });

        // Setup model
        TosaReference::ModelRunnerImpl runner;
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));
        TOSA_RETURN_ON_ERROR(runner.setInput(input1->GetName(), client_input1.data, client_input1.size));
        TOSA_RETURN_ON_ERROR(runner.setInput(input2->GetName(), client_input2.data, client_input2.size));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(runner.getOutput(output->GetName(), client_output.data, client_output.size));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_intdiv(tosa_tensor_t client_input1, tosa_tensor_t client_input2, tosa_tensor_t client_output)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        tosa::TosaSerializationTensor* input1 = translate_client_tensor(client_input1, "input1");
        tosa::TosaSerializationTensor* input2 = translate_client_tensor(client_input2, "input2");
        tosa::TosaSerializationTensor* output = translate_client_tensor(client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_INTDIV, tosa::Attribute::Attribute_NONE, &attr,
                                                      { input1->GetName(), input2->GetName() }, { output->GetName() });

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("intdiv", { op }, { input1, input2, output },
                                                { input1->GetName(), input2->GetName() }, { output->GetName() });

        // Setup model
        TosaReference::ModelRunnerImpl runner;
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));
        TOSA_RETURN_ON_ERROR(runner.setInput(input1->GetName(), client_input1.data, client_input1.size));
        TOSA_RETURN_ON_ERROR(runner.setInput(input2->GetName(), client_input2.data, client_input2.size));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(runner.getOutput(output->GetName(), client_output.data, client_output.size));

        return tosa_status_valid;
    }

    tosa_status_t
        tosa_run_logical_and(tosa_tensor_t client_input1, tosa_tensor_t client_input2, tosa_tensor_t client_output)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        tosa::TosaSerializationTensor* input1 = translate_client_tensor(client_input1, "input1");
        tosa::TosaSerializationTensor* input2 = translate_client_tensor(client_input2, "input2");
        tosa::TosaSerializationTensor* output = translate_client_tensor(client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_LOGICAL_AND, tosa::Attribute::Attribute_NONE, &attr,
                                                      { input1->GetName(), input2->GetName() }, { output->GetName() });

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("logical_and", { op }, { input1, input2, output },
                                                { input1->GetName(), input2->GetName() }, { output->GetName() });

        // Setup model
        TosaReference::ModelRunnerImpl runner;
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));
        TOSA_RETURN_ON_ERROR(runner.setInput(input1->GetName(), client_input1.data, client_input1.size));
        TOSA_RETURN_ON_ERROR(runner.setInput(input2->GetName(), client_input2.data, client_input2.size));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(runner.getOutput(output->GetName(), client_output.data, client_output.size));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_logical_left_shift(tosa_tensor_t client_input1,
                                              tosa_tensor_t client_input2,
                                              tosa_tensor_t client_output)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        tosa::TosaSerializationTensor* input1 = translate_client_tensor(client_input1, "input1");
        tosa::TosaSerializationTensor* input2 = translate_client_tensor(client_input2, "input2");
        tosa::TosaSerializationTensor* output = translate_client_tensor(client_output, "output");

        // Create operator
        auto op =
            new tosa::TosaSerializationOperator(tosa::Op::Op_LOGICAL_LEFT_SHIFT, tosa::Attribute::Attribute_NONE, &attr,
                                                { input1->GetName(), input2->GetName() }, { output->GetName() });

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("logical_left_shift", { op }, { input1, input2, output },
                                                { input1->GetName(), input2->GetName() }, { output->GetName() });

        // Setup model
        TosaReference::ModelRunnerImpl runner;
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));
        TOSA_RETURN_ON_ERROR(runner.setInput(input1->GetName(), client_input1.data, client_input1.size));
        TOSA_RETURN_ON_ERROR(runner.setInput(input2->GetName(), client_input2.data, client_input2.size));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(runner.getOutput(output->GetName(), client_output.data, client_output.size));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_logical_right_shift(tosa_tensor_t client_input1,
                                               tosa_tensor_t client_input2,
                                               tosa_tensor_t client_output)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        tosa::TosaSerializationTensor* input1 = translate_client_tensor(client_input1, "input1");
        tosa::TosaSerializationTensor* input2 = translate_client_tensor(client_input2, "input2");
        tosa::TosaSerializationTensor* output = translate_client_tensor(client_output, "output");

        // Create operator
        auto op =
            new tosa::TosaSerializationOperator(tosa::Op::Op_LOGICAL_RIGHT_SHIFT, tosa::Attribute::Attribute_NONE,
                                                &attr, { input1->GetName(), input2->GetName() }, { output->GetName() });

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("logical_right_shift", { op }, { input1, input2, output },
                                                { input1->GetName(), input2->GetName() }, { output->GetName() });

        // Setup model
        TosaReference::ModelRunnerImpl runner;
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));
        TOSA_RETURN_ON_ERROR(runner.setInput(input1->GetName(), client_input1.data, client_input1.size));
        TOSA_RETURN_ON_ERROR(runner.setInput(input2->GetName(), client_input2.data, client_input2.size));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(runner.getOutput(output->GetName(), client_output.data, client_output.size));

        return tosa_status_valid;
    }

    tosa_status_t
        tosa_run_logical_or(tosa_tensor_t client_input1, tosa_tensor_t client_input2, tosa_tensor_t client_output)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        tosa::TosaSerializationTensor* input1 = translate_client_tensor(client_input1, "input1");
        tosa::TosaSerializationTensor* input2 = translate_client_tensor(client_input2, "input2");
        tosa::TosaSerializationTensor* output = translate_client_tensor(client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_LOGICAL_OR, tosa::Attribute::Attribute_NONE, &attr,
                                                      { input1->GetName(), input2->GetName() }, { output->GetName() });

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("logical_or", { op }, { input1, input2, output },
                                                { input1->GetName(), input2->GetName() }, { output->GetName() });

        // Setup model
        TosaReference::ModelRunnerImpl runner;
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));
        TOSA_RETURN_ON_ERROR(runner.setInput(input1->GetName(), client_input1.data, client_input1.size));
        TOSA_RETURN_ON_ERROR(runner.setInput(input2->GetName(), client_input2.data, client_input2.size));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(runner.getOutput(output->GetName(), client_output.data, client_output.size));

        return tosa_status_valid;
    }

    tosa_status_t
        tosa_run_logical_xor(tosa_tensor_t client_input1, tosa_tensor_t client_input2, tosa_tensor_t client_output)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        tosa::TosaSerializationTensor* input1 = translate_client_tensor(client_input1, "input1");
        tosa::TosaSerializationTensor* input2 = translate_client_tensor(client_input2, "input2");
        tosa::TosaSerializationTensor* output = translate_client_tensor(client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_LOGICAL_XOR, tosa::Attribute::Attribute_NONE, &attr,
                                                      { input1->GetName(), input2->GetName() }, { output->GetName() });

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("logical_xor", { op }, { input1, input2, output },
                                                { input1->GetName(), input2->GetName() }, { output->GetName() });

        // Setup model
        TosaReference::ModelRunnerImpl runner;
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));
        TOSA_RETURN_ON_ERROR(runner.setInput(input1->GetName(), client_input1.data, client_input1.size));
        TOSA_RETURN_ON_ERROR(runner.setInput(input2->GetName(), client_input2.data, client_input2.size));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(runner.getOutput(output->GetName(), client_output.data, client_output.size));

        return tosa_status_valid;
    }

    tosa_status_t
        tosa_run_maximum(tosa_tensor_t client_input1, tosa_tensor_t client_input2, tosa_tensor_t client_output)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        tosa::TosaSerializationTensor* input1 = translate_client_tensor(client_input1, "input1");
        tosa::TosaSerializationTensor* input2 = translate_client_tensor(client_input2, "input2");
        tosa::TosaSerializationTensor* output = translate_client_tensor(client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_MAXIMUM, tosa::Attribute::Attribute_NONE, &attr,
                                                      { input1->GetName(), input2->GetName() }, { output->GetName() });

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("maximum", { op }, { input1, input2, output },
                                                { input1->GetName(), input2->GetName() }, { output->GetName() });

        // Setup model
        TosaReference::ModelRunnerImpl runner;
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));
        TOSA_RETURN_ON_ERROR(runner.setInput(input1->GetName(), client_input1.data, client_input1.size));
        TOSA_RETURN_ON_ERROR(runner.setInput(input2->GetName(), client_input2.data, client_input2.size));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(runner.getOutput(output->GetName(), client_output.data, client_output.size));

        return tosa_status_valid;
    }

    tosa_status_t
        tosa_run_minimum(tosa_tensor_t client_input1, tosa_tensor_t client_input2, tosa_tensor_t client_output)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        tosa::TosaSerializationTensor* input1 = translate_client_tensor(client_input1, "input1");
        tosa::TosaSerializationTensor* input2 = translate_client_tensor(client_input2, "input2");
        tosa::TosaSerializationTensor* output = translate_client_tensor(client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_MINIMUM, tosa::Attribute::Attribute_NONE, &attr,
                                                      { input1->GetName(), input2->GetName() }, { output->GetName() });

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("minimum", { op }, { input1, input2, output },
                                                { input1->GetName(), input2->GetName() }, { output->GetName() });

        // Setup model
        TosaReference::ModelRunnerImpl runner;
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));
        TOSA_RETURN_ON_ERROR(runner.setInput(input1->GetName(), client_input1.data, client_input1.size));
        TOSA_RETURN_ON_ERROR(runner.setInput(input2->GetName(), client_input2.data, client_input2.size));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(runner.getOutput(output->GetName(), client_output.data, client_output.size));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_mul(tosa_tensor_t client_input1,
                               tosa_tensor_t client_input2,
                               const uint8_t client_shift,
                               tosa_tensor_t client_output)
    {
        // Create operator attributes
        const int32_t shift = client_shift;
        TosaMulAttribute attr(shift);

        // Create tensors
        tosa::TosaSerializationTensor* input1 = translate_client_tensor(client_input1, "input1");
        tosa::TosaSerializationTensor* input2 = translate_client_tensor(client_input2, "input2");
        tosa::TosaSerializationTensor* output = translate_client_tensor(client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_MUL, tosa::Attribute::Attribute_MulAttribute, &attr,
                                                      { input1->GetName(), input2->GetName() }, { output->GetName() });

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("mul", { op }, { input1, input2, output },
                                                { input1->GetName(), input2->GetName() }, { output->GetName() });

        // Setup model
        TosaReference::ModelRunnerImpl runner;
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));
        TOSA_RETURN_ON_ERROR(runner.setInput(input1->GetName(), client_input1.data, client_input1.size));
        TOSA_RETURN_ON_ERROR(runner.setInput(input2->GetName(), client_input2.data, client_input2.size));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(runner.getOutput(output->GetName(), client_output.data, client_output.size));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_pow(tosa_tensor_t client_input1, tosa_tensor_t client_input2, tosa_tensor_t client_output)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        tosa::TosaSerializationTensor* input1 = translate_client_tensor(client_input1, "input1");
        tosa::TosaSerializationTensor* input2 = translate_client_tensor(client_input2, "input2");
        tosa::TosaSerializationTensor* output = translate_client_tensor(client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_POW, tosa::Attribute::Attribute_NONE, &attr,
                                                      { input1->GetName(), input2->GetName() }, { output->GetName() });

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("pow", { op }, { input1, input2, output },
                                                { input1->GetName(), input2->GetName() }, { output->GetName() });

        // Setup model
        TosaReference::ModelRunnerImpl runner;
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));
        TOSA_RETURN_ON_ERROR(runner.setInput(input1->GetName(), client_input1.data, client_input1.size));
        TOSA_RETURN_ON_ERROR(runner.setInput(input2->GetName(), client_input2.data, client_input2.size));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(runner.getOutput(output->GetName(), client_output.data, client_output.size));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_sub(tosa_tensor_t client_input1, tosa_tensor_t client_input2, tosa_tensor_t client_output)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        tosa::TosaSerializationTensor* input1 = translate_client_tensor(client_input1, "input1");
        tosa::TosaSerializationTensor* input2 = translate_client_tensor(client_input2, "input2");
        tosa::TosaSerializationTensor* output = translate_client_tensor(client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_SUB, tosa::Attribute::Attribute_NONE, &attr,
                                                      { input1->GetName(), input2->GetName() }, { output->GetName() });

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("sub", { op }, { input1, input2, output },
                                                { input1->GetName(), input2->GetName() }, { output->GetName() });

        // Setup model
        TosaReference::ModelRunnerImpl runner;
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));
        TOSA_RETURN_ON_ERROR(runner.setInput(input1->GetName(), client_input1.data, client_input1.size));
        TOSA_RETURN_ON_ERROR(runner.setInput(input2->GetName(), client_input2.data, client_input2.size));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(runner.getOutput(output->GetName(), client_output.data, client_output.size));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_table(tosa_tensor_t client_input,
                                 const int32_t client_table_len,
                                 const int16_t client_table[],
                                 tosa_tensor_t client_output)
    {
        // Create operator attributes
        const std::vector<int16_t> table(&client_table[0], &client_table[0] + client_table_len);
        TosaTableAttribute attr(table);

        // Create tensors
        tosa::TosaSerializationTensor* input  = translate_client_tensor(client_input, "input");
        tosa::TosaSerializationTensor* output = translate_client_tensor(client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_TABLE, tosa::Attribute::Attribute_TableAttribute,
                                                      &attr, { input->GetName() }, { output->GetName() });

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("table", { op }, { input, output }, { input->GetName() },
                                                { output->GetName() });

        // Setup model
        TosaReference::ModelRunnerImpl runner;
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));
        TOSA_RETURN_ON_ERROR(runner.setInput(input->GetName(), client_input.data, client_input.size));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(runner.getOutput(output->GetName(), client_output.data, client_output.size));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_abs(tosa_tensor_t client_input1, tosa_tensor_t client_output)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        tosa::TosaSerializationTensor* input1 = translate_client_tensor(client_input1, "input1");
        tosa::TosaSerializationTensor* output = translate_client_tensor(client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_ABS, tosa::Attribute::Attribute_NONE, &attr,
                                                      { input1->GetName() }, { output->GetName() });

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("abs", { op }, { input1, output }, { input1->GetName() },
                                                { output->GetName() });

        // Setup model
        TosaReference::ModelRunnerImpl runner;
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));
        TOSA_RETURN_ON_ERROR(runner.setInput(input1->GetName(), client_input1.data, client_input1.size));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(runner.getOutput(output->GetName(), client_output.data, client_output.size));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_bitwise_not(tosa_tensor_t client_input1, tosa_tensor_t client_output)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        tosa::TosaSerializationTensor* input1 = translate_client_tensor(client_input1, "input1");
        tosa::TosaSerializationTensor* output = translate_client_tensor(client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_BITWISE_NOT, tosa::Attribute::Attribute_NONE, &attr,
                                                      { input1->GetName() }, { output->GetName() });

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("bitwise_not", { op }, { input1, output }, { input1->GetName() },
                                                { output->GetName() });

        // Setup model
        TosaReference::ModelRunnerImpl runner;
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));
        TOSA_RETURN_ON_ERROR(runner.setInput(input1->GetName(), client_input1.data, client_input1.size));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(runner.getOutput(output->GetName(), client_output.data, client_output.size));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_ceil(tosa_tensor_t client_input1, tosa_tensor_t client_output)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        tosa::TosaSerializationTensor* input1 = translate_client_tensor(client_input1, "input1");
        tosa::TosaSerializationTensor* output = translate_client_tensor(client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_CEIL, tosa::Attribute::Attribute_NONE, &attr,
                                                      { input1->GetName() }, { output->GetName() });

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("ceil", { op }, { input1, output }, { input1->GetName() },
                                                { output->GetName() });

        // Setup model
        TosaReference::ModelRunnerImpl runner;
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));
        TOSA_RETURN_ON_ERROR(runner.setInput(input1->GetName(), client_input1.data, client_input1.size));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(runner.getOutput(output->GetName(), client_output.data, client_output.size));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_clz(tosa_tensor_t client_input1, tosa_tensor_t client_output)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        tosa::TosaSerializationTensor* input1 = translate_client_tensor(client_input1, "input1");
        tosa::TosaSerializationTensor* output = translate_client_tensor(client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_CLZ, tosa::Attribute::Attribute_NONE, &attr,
                                                      { input1->GetName() }, { output->GetName() });

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("clz", { op }, { input1, output }, { input1->GetName() },
                                                { output->GetName() });

        // Setup model
        TosaReference::ModelRunnerImpl runner;
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));
        TOSA_RETURN_ON_ERROR(runner.setInput(input1->GetName(), client_input1.data, client_input1.size));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(runner.getOutput(output->GetName(), client_output.data, client_output.size));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_exp(tosa_tensor_t client_input1, tosa_tensor_t client_output)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        tosa::TosaSerializationTensor* input1 = translate_client_tensor(client_input1, "input1");
        tosa::TosaSerializationTensor* output = translate_client_tensor(client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_EXP, tosa::Attribute::Attribute_NONE, &attr,
                                                      { input1->GetName() }, { output->GetName() });

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("exp", { op }, { input1, output }, { input1->GetName() },
                                                { output->GetName() });

        // Setup model
        TosaReference::ModelRunnerImpl runner;
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));
        TOSA_RETURN_ON_ERROR(runner.setInput(input1->GetName(), client_input1.data, client_input1.size));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(runner.getOutput(output->GetName(), client_output.data, client_output.size));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_floor(tosa_tensor_t client_input1, tosa_tensor_t client_output)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        tosa::TosaSerializationTensor* input1 = translate_client_tensor(client_input1, "input1");
        tosa::TosaSerializationTensor* output = translate_client_tensor(client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_FLOOR, tosa::Attribute::Attribute_NONE, &attr,
                                                      { input1->GetName() }, { output->GetName() });

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("floor", { op }, { input1, output }, { input1->GetName() },
                                                { output->GetName() });

        // Setup model
        TosaReference::ModelRunnerImpl runner;
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));
        TOSA_RETURN_ON_ERROR(runner.setInput(input1->GetName(), client_input1.data, client_input1.size));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(runner.getOutput(output->GetName(), client_output.data, client_output.size));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_log(tosa_tensor_t client_input1, tosa_tensor_t client_output)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        tosa::TosaSerializationTensor* input1 = translate_client_tensor(client_input1, "input1");
        tosa::TosaSerializationTensor* output = translate_client_tensor(client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_LOG, tosa::Attribute::Attribute_NONE, &attr,
                                                      { input1->GetName() }, { output->GetName() });

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("log", { op }, { input1, output }, { input1->GetName() },
                                                { output->GetName() });

        // Setup model
        TosaReference::ModelRunnerImpl runner;
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));
        TOSA_RETURN_ON_ERROR(runner.setInput(input1->GetName(), client_input1.data, client_input1.size));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(runner.getOutput(output->GetName(), client_output.data, client_output.size));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_logical_not(tosa_tensor_t client_input1, tosa_tensor_t client_output)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        tosa::TosaSerializationTensor* input1 = translate_client_tensor(client_input1, "input1");
        tosa::TosaSerializationTensor* output = translate_client_tensor(client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_LOGICAL_NOT, tosa::Attribute::Attribute_NONE, &attr,
                                                      { input1->GetName() }, { output->GetName() });

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("logical_not", { op }, { input1, output }, { input1->GetName() },
                                                { output->GetName() });

        // Setup model
        TosaReference::ModelRunnerImpl runner;
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));
        TOSA_RETURN_ON_ERROR(runner.setInput(input1->GetName(), client_input1.data, client_input1.size));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(runner.getOutput(output->GetName(), client_output.data, client_output.size));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_negate(tosa_tensor_t client_input1,
                                  const int32_t client_input1_zp,
                                  const int32_t client_output_zp,
                                  tosa_tensor_t client_output)
    {
        // Create operator attributes
        const int32_t input1_zp = client_input1_zp;
        const int32_t output_zp = client_output_zp;
        TosaNegateAttribute attr(input1_zp, output_zp);

        // Create tensors
        tosa::TosaSerializationTensor* input1 = translate_client_tensor(client_input1, "input1");
        tosa::TosaSerializationTensor* output = translate_client_tensor(client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_NEGATE, tosa::Attribute::Attribute_NegateAttribute,
                                                      &attr, { input1->GetName() }, { output->GetName() });

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("negate", { op }, { input1, output }, { input1->GetName() },
                                                { output->GetName() });

        // Setup model
        TosaReference::ModelRunnerImpl runner;
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));
        TOSA_RETURN_ON_ERROR(runner.setInput(input1->GetName(), client_input1.data, client_input1.size));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(runner.getOutput(output->GetName(), client_output.data, client_output.size));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_reciprocal(tosa_tensor_t client_input1, tosa_tensor_t client_output)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        tosa::TosaSerializationTensor* input1 = translate_client_tensor(client_input1, "input1");
        tosa::TosaSerializationTensor* output = translate_client_tensor(client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_RECIPROCAL, tosa::Attribute::Attribute_NONE, &attr,
                                                      { input1->GetName() }, { output->GetName() });

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("reciprocal", { op }, { input1, output }, { input1->GetName() },
                                                { output->GetName() });

        // Setup model
        TosaReference::ModelRunnerImpl runner;
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));
        TOSA_RETURN_ON_ERROR(runner.setInput(input1->GetName(), client_input1.data, client_input1.size));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(runner.getOutput(output->GetName(), client_output.data, client_output.size));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_rsqrt(tosa_tensor_t client_input1, tosa_tensor_t client_output)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        tosa::TosaSerializationTensor* input1 = translate_client_tensor(client_input1, "input1");
        tosa::TosaSerializationTensor* output = translate_client_tensor(client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_RSQRT, tosa::Attribute::Attribute_NONE, &attr,
                                                      { input1->GetName() }, { output->GetName() });

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("rsqrt", { op }, { input1, output }, { input1->GetName() },
                                                { output->GetName() });

        // Setup model
        TosaReference::ModelRunnerImpl runner;
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));
        TOSA_RETURN_ON_ERROR(runner.setInput(input1->GetName(), client_input1.data, client_input1.size));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(runner.getOutput(output->GetName(), client_output.data, client_output.size));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_select(tosa_tensor_t client_input1,
                                  tosa_tensor_t client_input2,
                                  tosa_tensor_t client_input3,
                                  tosa_tensor_t client_output)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        tosa::TosaSerializationTensor* input1 = translate_client_tensor(client_input1, "input1");
        tosa::TosaSerializationTensor* input2 = translate_client_tensor(client_input2, "input2");
        tosa::TosaSerializationTensor* input3 = translate_client_tensor(client_input3, "input3");
        tosa::TosaSerializationTensor* output = translate_client_tensor(client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_SELECT, tosa::Attribute::Attribute_NONE, &attr,
                                                      { input1->GetName(), input2->GetName(), input3->GetName() },
                                                      { output->GetName() });

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("select", { op }, { input1, input2, input3, output },
                                                { input1->GetName(), input2->GetName(), input3->GetName() },
                                                { output->GetName() });

        // Setup model
        TosaReference::ModelRunnerImpl runner;
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));
        TOSA_RETURN_ON_ERROR(runner.setInput(input1->GetName(), client_input1.data, client_input1.size));
        TOSA_RETURN_ON_ERROR(runner.setInput(input2->GetName(), client_input2.data, client_input2.size));
        TOSA_RETURN_ON_ERROR(runner.setInput(input3->GetName(), client_input3.data, client_input3.size));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(runner.getOutput(output->GetName(), client_output.data, client_output.size));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_equal(tosa_tensor_t client_input1, tosa_tensor_t client_input2, tosa_tensor_t client_output)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        tosa::TosaSerializationTensor* input1 = translate_client_tensor(client_input1, "input1");
        tosa::TosaSerializationTensor* input2 = translate_client_tensor(client_input2, "input2");
        tosa::TosaSerializationTensor* output = translate_client_tensor(client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_EQUAL, tosa::Attribute::Attribute_NONE, &attr,
                                                      { input1->GetName(), input2->GetName() }, { output->GetName() });

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("equal", { op }, { input1, input2, output },
                                                { input1->GetName(), input2->GetName() }, { output->GetName() });

        // Setup model
        TosaReference::ModelRunnerImpl runner;
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));
        TOSA_RETURN_ON_ERROR(runner.setInput(input1->GetName(), client_input1.data, client_input1.size));
        TOSA_RETURN_ON_ERROR(runner.setInput(input2->GetName(), client_input2.data, client_input2.size));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(runner.getOutput(output->GetName(), client_output.data, client_output.size));

        return tosa_status_valid;
    }

    tosa_status_t
        tosa_run_greater(tosa_tensor_t client_input1, tosa_tensor_t client_input2, tosa_tensor_t client_output)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        tosa::TosaSerializationTensor* input1 = translate_client_tensor(client_input1, "input1");
        tosa::TosaSerializationTensor* input2 = translate_client_tensor(client_input2, "input2");
        tosa::TosaSerializationTensor* output = translate_client_tensor(client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_GREATER, tosa::Attribute::Attribute_NONE, &attr,
                                                      { input1->GetName(), input2->GetName() }, { output->GetName() });

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("greater", { op }, { input1, input2, output },
                                                { input1->GetName(), input2->GetName() }, { output->GetName() });

        // Setup model
        TosaReference::ModelRunnerImpl runner;
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));
        TOSA_RETURN_ON_ERROR(runner.setInput(input1->GetName(), client_input1.data, client_input1.size));
        TOSA_RETURN_ON_ERROR(runner.setInput(input2->GetName(), client_input2.data, client_input2.size));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(runner.getOutput(output->GetName(), client_output.data, client_output.size));

        return tosa_status_valid;
    }

    tosa_status_t
        tosa_run_greater_equal(tosa_tensor_t client_input1, tosa_tensor_t client_input2, tosa_tensor_t client_output)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        tosa::TosaSerializationTensor* input1 = translate_client_tensor(client_input1, "input1");
        tosa::TosaSerializationTensor* input2 = translate_client_tensor(client_input2, "input2");
        tosa::TosaSerializationTensor* output = translate_client_tensor(client_output, "output");

        // Create operator
        auto op =
            new tosa::TosaSerializationOperator(tosa::Op::Op_GREATER_EQUAL, tosa::Attribute::Attribute_NONE, &attr,
                                                { input1->GetName(), input2->GetName() }, { output->GetName() });

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("greater_equal", { op }, { input1, input2, output },
                                                { input1->GetName(), input2->GetName() }, { output->GetName() });

        // Setup model
        TosaReference::ModelRunnerImpl runner;
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));
        TOSA_RETURN_ON_ERROR(runner.setInput(input1->GetName(), client_input1.data, client_input1.size));
        TOSA_RETURN_ON_ERROR(runner.setInput(input2->GetName(), client_input2.data, client_input2.size));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(runner.getOutput(output->GetName(), client_output.data, client_output.size));

        return tosa_status_valid;
    }

    tosa_status_t
        tosa_run_reduce_all(tosa_tensor_t client_input, const int32_t client_axis, tosa_tensor_t client_output)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        tosa::TosaSerializationTensor* input  = translate_client_tensor(client_input, "input");
        tosa::TosaSerializationTensor* output = translate_client_tensor(client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_REDUCE_ALL, tosa::Attribute::Attribute_NONE, &attr,
                                                      { input->GetName() }, { output->GetName() });

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("reduce_all", { op }, { input, output }, { input->GetName() },
                                                { output->GetName() });

        // Setup model
        TosaReference::ModelRunnerImpl runner;
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));
        TOSA_RETURN_ON_ERROR(runner.setInput(input->GetName(), client_input.data, client_input.size));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(runner.getOutput(output->GetName(), client_output.data, client_output.size));

        return tosa_status_valid;
    }

    tosa_status_t
        tosa_run_reduce_any(tosa_tensor_t client_input, const int32_t client_axis, tosa_tensor_t client_output)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        tosa::TosaSerializationTensor* input  = translate_client_tensor(client_input, "input");
        tosa::TosaSerializationTensor* output = translate_client_tensor(client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_REDUCE_ANY, tosa::Attribute::Attribute_NONE, &attr,
                                                      { input->GetName() }, { output->GetName() });

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("reduce_any", { op }, { input, output }, { input->GetName() },
                                                { output->GetName() });

        // Setup model
        TosaReference::ModelRunnerImpl runner;
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));
        TOSA_RETURN_ON_ERROR(runner.setInput(input->GetName(), client_input.data, client_input.size));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(runner.getOutput(output->GetName(), client_output.data, client_output.size));

        return tosa_status_valid;
    }

    tosa_status_t
        tosa_run_reduce_max(tosa_tensor_t client_input, const int32_t client_axis, tosa_tensor_t client_output)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        tosa::TosaSerializationTensor* input  = translate_client_tensor(client_input, "input");
        tosa::TosaSerializationTensor* output = translate_client_tensor(client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_REDUCE_MAX, tosa::Attribute::Attribute_NONE, &attr,
                                                      { input->GetName() }, { output->GetName() });

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("reduce_max", { op }, { input, output }, { input->GetName() },
                                                { output->GetName() });

        // Setup model
        TosaReference::ModelRunnerImpl runner;
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));
        TOSA_RETURN_ON_ERROR(runner.setInput(input->GetName(), client_input.data, client_input.size));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(runner.getOutput(output->GetName(), client_output.data, client_output.size));

        return tosa_status_valid;
    }

    tosa_status_t
        tosa_run_reduce_min(tosa_tensor_t client_input, const int32_t client_axis, tosa_tensor_t client_output)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        tosa::TosaSerializationTensor* input  = translate_client_tensor(client_input, "input");
        tosa::TosaSerializationTensor* output = translate_client_tensor(client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_REDUCE_MIN, tosa::Attribute::Attribute_NONE, &attr,
                                                      { input->GetName() }, { output->GetName() });

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("reduce_min", { op }, { input, output }, { input->GetName() },
                                                { output->GetName() });

        // Setup model
        TosaReference::ModelRunnerImpl runner;
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));
        TOSA_RETURN_ON_ERROR(runner.setInput(input->GetName(), client_input.data, client_input.size));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(runner.getOutput(output->GetName(), client_output.data, client_output.size));

        return tosa_status_valid;
    }

    tosa_status_t
        tosa_run_reduce_product(tosa_tensor_t client_input, const int32_t client_axis, tosa_tensor_t client_output)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        tosa::TosaSerializationTensor* input  = translate_client_tensor(client_input, "input");
        tosa::TosaSerializationTensor* output = translate_client_tensor(client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_REDUCE_PRODUCT, tosa::Attribute::Attribute_NONE,
                                                      &attr, { input->GetName() }, { output->GetName() });

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("reduce_product", { op }, { input, output }, { input->GetName() },
                                                { output->GetName() });

        // Setup model
        TosaReference::ModelRunnerImpl runner;
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));
        TOSA_RETURN_ON_ERROR(runner.setInput(input->GetName(), client_input.data, client_input.size));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(runner.getOutput(output->GetName(), client_output.data, client_output.size));

        return tosa_status_valid;
    }

    tosa_status_t
        tosa_run_reduce_sum(tosa_tensor_t client_input, const int32_t client_axis, tosa_tensor_t client_output)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        tosa::TosaSerializationTensor* input  = translate_client_tensor(client_input, "input");
        tosa::TosaSerializationTensor* output = translate_client_tensor(client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_REDUCE_SUM, tosa::Attribute::Attribute_NONE, &attr,
                                                      { input->GetName() }, { output->GetName() });

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("reduce_sum", { op }, { input, output }, { input->GetName() },
                                                { output->GetName() });

        // Setup model
        TosaReference::ModelRunnerImpl runner;
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));
        TOSA_RETURN_ON_ERROR(runner.setInput(input->GetName(), client_input.data, client_input.size));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(runner.getOutput(output->GetName(), client_output.data, client_output.size));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_concat(tosa_tensor_t client_input1, const int32_t client_axis, tosa_tensor_t client_output)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        tosa::TosaSerializationTensor* input1 = translate_client_tensor(client_input1, "input1");
        tosa::TosaSerializationTensor* output = translate_client_tensor(client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_CONCAT, tosa::Attribute::Attribute_NONE, &attr,
                                                      { input1->GetName() }, { output->GetName() });

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("concat", { op }, { input1, output }, { input1->GetName() },
                                                { output->GetName() });

        // Setup model
        TosaReference::ModelRunnerImpl runner;
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));
        TOSA_RETURN_ON_ERROR(runner.setInput(input1->GetName(), client_input1.data, client_input1.size));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(runner.getOutput(output->GetName(), client_output.data, client_output.size));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_pad(tosa_tensor_t client_input1,
                               const int32_t client_padding_len,
                               const int32_t client_padding[],
                               const int32_t client_pad_const_int,
                               const float client_pad_const_fp,
                               tosa_tensor_t client_output)
    {
        // Create operator attributes
        const std::vector<int32_t> padding(&client_padding[0], &client_padding[0] + client_padding_len);
        const int32_t pad_const_int = client_pad_const_int;
        const float pad_const_fp    = client_pad_const_fp;
        TosaPadAttribute attr(padding, pad_const_int, pad_const_fp);

        // Create tensors
        tosa::TosaSerializationTensor* input1 = translate_client_tensor(client_input1, "input1");
        tosa::TosaSerializationTensor* output = translate_client_tensor(client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_PAD, tosa::Attribute::Attribute_PadAttribute, &attr,
                                                      { input1->GetName() }, { output->GetName() });

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("pad", { op }, { input1, output }, { input1->GetName() },
                                                { output->GetName() });

        // Setup model
        TosaReference::ModelRunnerImpl runner;
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));
        TOSA_RETURN_ON_ERROR(runner.setInput(input1->GetName(), client_input1.data, client_input1.size));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(runner.getOutput(output->GetName(), client_output.data, client_output.size));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_reshape(tosa_tensor_t client_input1,
                                   const int32_t client_new_shape_len,
                                   const int32_t client_new_shape[],
                                   tosa_tensor_t client_output)
    {
        // Create operator attributes
        const std::vector<int32_t> new_shape(&client_new_shape[0], &client_new_shape[0] + client_new_shape_len);
        TosaReshapeAttribute attr(new_shape);

        // Create tensors
        tosa::TosaSerializationTensor* input1 = translate_client_tensor(client_input1, "input1");
        tosa::TosaSerializationTensor* output = translate_client_tensor(client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_RESHAPE, tosa::Attribute::Attribute_ReshapeAttribute,
                                                      &attr, { input1->GetName() }, { output->GetName() });

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("reshape", { op }, { input1, output }, { input1->GetName() },
                                                { output->GetName() });

        // Setup model
        TosaReference::ModelRunnerImpl runner;
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));
        TOSA_RETURN_ON_ERROR(runner.setInput(input1->GetName(), client_input1.data, client_input1.size));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(runner.getOutput(output->GetName(), client_output.data, client_output.size));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_reverse(tosa_tensor_t client_input, const int32_t client_axis, tosa_tensor_t client_output)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        tosa::TosaSerializationTensor* input  = translate_client_tensor(client_input, "input");
        tosa::TosaSerializationTensor* output = translate_client_tensor(client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_REVERSE, tosa::Attribute::Attribute_NONE, &attr,
                                                      { input->GetName() }, { output->GetName() });

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("reverse", { op }, { input, output }, { input->GetName() },
                                                { output->GetName() });

        // Setup model
        TosaReference::ModelRunnerImpl runner;
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));
        TOSA_RETURN_ON_ERROR(runner.setInput(input->GetName(), client_input.data, client_input.size));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(runner.getOutput(output->GetName(), client_output.data, client_output.size));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_slice(tosa_tensor_t client_input1,
                                 const int32_t client_start_len,
                                 const int32_t client_start[],
                                 const int32_t client_size_len,
                                 const int32_t client_size[],
                                 tosa_tensor_t client_output)
    {
        // Create operator attributes
        const std::vector<int32_t> start(&client_start[0], &client_start[0] + client_start_len);
        const std::vector<int32_t> size(&client_size[0], &client_size[0] + client_size_len);
        TosaSliceAttribute attr(start, size);

        // Create tensors
        tosa::TosaSerializationTensor* input1 = translate_client_tensor(client_input1, "input1");
        tosa::TosaSerializationTensor* output = translate_client_tensor(client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_SLICE, tosa::Attribute::Attribute_SliceAttribute,
                                                      &attr, { input1->GetName() }, { output->GetName() });

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("slice", { op }, { input1, output }, { input1->GetName() },
                                                { output->GetName() });

        // Setup model
        TosaReference::ModelRunnerImpl runner;
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));
        TOSA_RETURN_ON_ERROR(runner.setInput(input1->GetName(), client_input1.data, client_input1.size));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(runner.getOutput(output->GetName(), client_output.data, client_output.size));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_tile(tosa_tensor_t client_input1,
                                const int32_t client_multiplies_len,
                                const int32_t client_multiplies[],
                                const int32_t client_multiples_len,
                                const int32_t client_multiples[],
                                tosa_tensor_t client_output)
    {
        // Create operator attributes
        const std::vector<int32_t> multiples(&client_multiples[0], &client_multiples[0] + client_multiples_len);
        TosaTileAttribute attr(multiples);

        // Create tensors
        tosa::TosaSerializationTensor* input1 = translate_client_tensor(client_input1, "input1");
        tosa::TosaSerializationTensor* output = translate_client_tensor(client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_TILE, tosa::Attribute::Attribute_TileAttribute,
                                                      &attr, { input1->GetName() }, { output->GetName() });

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("tile", { op }, { input1, output }, { input1->GetName() },
                                                { output->GetName() });

        // Setup model
        TosaReference::ModelRunnerImpl runner;
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));
        TOSA_RETURN_ON_ERROR(runner.setInput(input1->GetName(), client_input1.data, client_input1.size));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(runner.getOutput(output->GetName(), client_output.data, client_output.size));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_transpose(tosa_tensor_t client_input1,
                                     const int32_t client_perms_len,
                                     const int32_t client_perms[],
                                     tosa_tensor_t client_output)
    {
        // Create operator attributes
        const std::vector<int32_t> perms(&client_perms[0], &client_perms[0] + client_perms_len);
        TosaTransposeAttribute attr(perms);

        // Create tensors
        tosa::TosaSerializationTensor* input1 = translate_client_tensor(client_input1, "input1");
        tosa::TosaSerializationTensor* output = translate_client_tensor(client_output, "output");

        // Create operator
        auto op =
            new tosa::TosaSerializationOperator(tosa::Op::Op_TRANSPOSE, tosa::Attribute::Attribute_TransposeAttribute,
                                                &attr, { input1->GetName() }, { output->GetName() });

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("transpose", { op }, { input1, output }, { input1->GetName() },
                                                { output->GetName() });

        // Setup model
        TosaReference::ModelRunnerImpl runner;
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));
        TOSA_RETURN_ON_ERROR(runner.setInput(input1->GetName(), client_input1.data, client_input1.size));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(runner.getOutput(output->GetName(), client_output.data, client_output.size));

        return tosa_status_valid;
    }

    tosa_status_t
        tosa_run_gather(tosa_tensor_t client_values, tosa_tensor_t client_indices, tosa_tensor_t client_output)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        tosa::TosaSerializationTensor* values  = translate_client_tensor(client_values, "values");
        tosa::TosaSerializationTensor* indices = translate_client_tensor(client_indices, "indices");
        tosa::TosaSerializationTensor* output  = translate_client_tensor(client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_GATHER, tosa::Attribute::Attribute_NONE, &attr,
                                                      { values->GetName(), indices->GetName() }, { output->GetName() });

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("gather", { op }, { values, indices, output },
                                                { values->GetName(), indices->GetName() }, { output->GetName() });

        // Setup model
        TosaReference::ModelRunnerImpl runner;
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));
        TOSA_RETURN_ON_ERROR(runner.setInput(values->GetName(), client_values.data, client_values.size));
        TOSA_RETURN_ON_ERROR(runner.setInput(indices->GetName(), client_indices.data, client_indices.size));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(runner.getOutput(output->GetName(), client_output.data, client_output.size));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_scatter(tosa_tensor_t client_values_in,
                                   tosa_tensor_t client_indices,
                                   tosa_tensor_t client_input,
                                   tosa_tensor_t client_values_out)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        tosa::TosaSerializationTensor* values_in  = translate_client_tensor(client_values_in, "values_in");
        tosa::TosaSerializationTensor* indices    = translate_client_tensor(client_indices, "indices");
        tosa::TosaSerializationTensor* input      = translate_client_tensor(client_input, "input");
        tosa::TosaSerializationTensor* values_out = translate_client_tensor(client_values_out, "values_out");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_SCATTER, tosa::Attribute::Attribute_NONE, &attr,
                                                      { values_in->GetName(), indices->GetName(), input->GetName() },
                                                      { values_out->GetName() });

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("scatter", { op }, { values_in, indices, input, values_out },
                                                { values_in->GetName(), indices->GetName(), input->GetName() },
                                                { values_out->GetName() });

        // Setup model
        TosaReference::ModelRunnerImpl runner;
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));
        TOSA_RETURN_ON_ERROR(runner.setInput(values_in->GetName(), client_values_in.data, client_values_in.size));
        TOSA_RETURN_ON_ERROR(runner.setInput(indices->GetName(), client_indices.data, client_indices.size));
        TOSA_RETURN_ON_ERROR(runner.setInput(input->GetName(), client_input.data, client_input.size));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(runner.getOutput(values_out->GetName(), client_values_out.data, client_values_out.size));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_resize(tosa_tensor_t client_input,
                                  const int16_t client_scale[4],
                                  const int16_t client_offset[2],
                                  const int16_t client_border[2],
                                  const tosa_mode_t client_mode,
                                  tosa_tensor_t client_output)
    {
        // Create operator attributes
        const std::vector<int16_t> scale(&client_scale[0], &client_scale[4]);
        const std::vector<int16_t> offset(&client_offset[0], &client_offset[2]);
        const std::vector<int16_t> border(&client_border[0], &client_border[2]);
        const ResizeMode mode = translate_client_tosa_mode(client_mode);
        TosaResizeAttribute attr(scale, offset, border, mode);

        // Create tensors
        tosa::TosaSerializationTensor* input  = translate_client_tensor(client_input, "input");
        tosa::TosaSerializationTensor* output = translate_client_tensor(client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_RESIZE, tosa::Attribute::Attribute_ResizeAttribute,
                                                      &attr, { input->GetName() }, { output->GetName() });

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("resize", { op }, { input, output }, { input->GetName() },
                                                { output->GetName() });

        // Setup model
        TosaReference::ModelRunnerImpl runner;
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));
        TOSA_RETURN_ON_ERROR(runner.setInput(input->GetName(), client_input.data, client_input.size));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(runner.getOutput(output->GetName(), client_output.data, client_output.size));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_cast(tosa_tensor_t client_input, tosa_tensor_t client_output)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        tosa::TosaSerializationTensor* input  = translate_client_tensor(client_input, "input");
        tosa::TosaSerializationTensor* output = translate_client_tensor(client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_CAST, tosa::Attribute::Attribute_NONE, &attr,
                                                      { input->GetName() }, { output->GetName() });

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("cast", { op }, { input, output }, { input->GetName() },
                                                { output->GetName() });

        // Setup model
        TosaReference::ModelRunnerImpl runner;
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));
        TOSA_RETURN_ON_ERROR(runner.setInput(input->GetName(), client_input.data, client_input.size));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(runner.getOutput(output->GetName(), client_output.data, client_output.size));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_rescale(tosa_tensor_t client_input,
                                   tosa_tensor_t client_output,
                                   const int32_t client_input_zp,
                                   const int32_t client_output_zp,
                                   const int32_t client_multiplier_len,
                                   const int32_t client_multiplier[],
                                   const int32_t client_shift_len,
                                   const uint8_t client_shift[],
                                   const bool client_scale32,
                                   const bool client_double_round,
                                   const bool client_per_channel)
    {
        // Create operator attributes
        const int32_t input_zp  = client_input_zp;
        const int32_t output_zp = client_output_zp;
        const std::vector<int32_t> multiplier(&client_multiplier[0], &client_multiplier[0] + client_multiplier_len);
        const std::vector<int32_t> shift(&client_shift[0], &client_shift[0] + client_shift_len);
        const bool scale32      = client_scale32;
        const bool double_round = client_double_round;
        const bool per_channel  = client_per_channel;
        TosaRescaleAttribute attr(input_zp, output_zp, multiplier, shift, scale32, double_round, per_channel);

        // Create tensors
        tosa::TosaSerializationTensor* input  = translate_client_tensor(client_input, "input");
        tosa::TosaSerializationTensor* output = translate_client_tensor(client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_RESCALE, tosa::Attribute::Attribute_RescaleAttribute,
                                                      &attr, { input->GetName() }, { output->GetName() });

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("rescale", { op }, { input, output }, { input->GetName() },
                                                { output->GetName() });

        // Setup model
        TosaReference::ModelRunnerImpl runner;
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));
        TOSA_RETURN_ON_ERROR(runner.setInput(input->GetName(), client_input.data, client_input.size));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(runner.getOutput(output->GetName(), client_output.data, client_output.size));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_identity(tosa_tensor_t client_input1, tosa_tensor_t client_output)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        tosa::TosaSerializationTensor* input1 = translate_client_tensor(client_input1, "input1");
        tosa::TosaSerializationTensor* output = translate_client_tensor(client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_IDENTITY, tosa::Attribute::Attribute_NONE, &attr,
                                                      { input1->GetName() }, { output->GetName() });

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("identity", { op }, { input1, output }, { input1->GetName() },
                                                { output->GetName() });

        // Setup model
        TosaReference::ModelRunnerImpl runner;
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));
        TOSA_RETURN_ON_ERROR(runner.setInput(input1->GetName(), client_input1.data, client_input1.size));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(runner.getOutput(output->GetName(), client_output.data, client_output.size));

        return tosa_status_valid;
    }

}    // extern "C"
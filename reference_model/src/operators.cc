
// Copyright (c) 2022-2023, ARM Limited.
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

#define TOSA_PROPAGATE_ERROR(status)                                                                                   \
    do                                                                                                                 \
    {                                                                                                                  \
        if (status != 0)                                                                                               \
        {                                                                                                              \
            return status;                                                                                             \
        }                                                                                                              \
    } while (false)

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
        case tosa_datatype_bf16_t:
            return tosa::DType::DType_BF16;
        case tosa_datatype_bool_t:
            return tosa::DType::DType_BOOL;
        case tosa_datatype_fp16_t:
            return tosa::DType::DType_FP16;
        case tosa_datatype_fp32_t:
            return tosa::DType::DType_FP32;
        case tosa_datatype_int16_t:
            return tosa::DType::DType_INT16;
        case tosa_datatype_int32_t:
            return tosa::DType::DType_INT32;
        case tosa_datatype_int48_t:
            return tosa::DType::DType_INT48;
        case tosa_datatype_int4_t:
            return tosa::DType::DType_INT4;
        case tosa_datatype_int8_t:
            return tosa::DType::DType_INT8;
        case tosa_datatype_uint16_t:
            return tosa::DType::DType_UINT16;
        case tosa_datatype_uint8_t:
            return tosa::DType::DType_UINT8;
        case tosa_datatype_shape_t:
            return tosa::DType::DType_SHAPE;
        default:
            return tosa::DType::DType_UNKNOWN;
    }
};

using TosaTensorInfo = std::pair<tosa::TosaSerializationTensor*, tosa_tensor_t*>;

tosa::TosaSerializationTensor* translate_client_tensor(tosa_tensor_t& tensor, const std::string& name)
{
    std::vector<int32_t> shape(tensor.shape, tensor.shape + tensor.num_dims);
    return new tosa::TosaSerializationTensor(name, shape, translate_client_datatype(tensor.data_type), {});
}

void addTensor(std::vector<TosaTensorInfo>& tensors, tosa_tensor_t& tensor, std::string tensorName)
{
    auto tensorDescr = translate_client_tensor(tensor, tensorName);
    tensors.push_back(std::make_pair(tensorDescr, &tensor));
}

int setInputTensors(TosaReference::ModelRunnerImpl& runner, std::vector<TosaTensorInfo>& inputTensors)
{
    for (const auto& [tensorDescr, tensorData] : inputTensors)
    {
        auto status = runner.setInput(tensorDescr->GetName(), tensorData->data, tensorData->size);
        TOSA_PROPAGATE_ERROR(status);
    }

    return 0;
}

int getOutputTensors(TosaReference::ModelRunnerImpl& runner, std::vector<TosaTensorInfo>& outputTensors)
{
    for (const auto& [tensorDescr, tensorData] : outputTensors)
    {
        auto status = runner.getOutput(tensorDescr->GetName(), tensorData->data, tensorData->size);
        TOSA_PROPAGATE_ERROR(status);
    }

    return 0;
}

std::vector<std::string> getTensorNames(std::vector<TosaTensorInfo>& tensors)
{
    std::vector<std::string> tensorNames;
    const auto mapping = [](const TosaTensorInfo& info) { return info.first->GetName(); };

    std::transform(tensors.cbegin(), tensors.cend(), std::back_inserter(tensorNames), mapping);
    return tensorNames;
}

std::vector<TosaSerializationTensor*> allTensors(std::vector<TosaTensorInfo>& inputTensors,
                                                 std::vector<TosaTensorInfo>& outputTensors)
{
    std::vector<TosaSerializationTensor*> result;
    const auto mapping = [](const TosaTensorInfo& info) { return info.first; };

    std::transform(inputTensors.cbegin(), inputTensors.cend(), std::back_inserter(result), mapping);
    std::transform(outputTensors.cbegin(), outputTensors.cend(), std::back_inserter(result), mapping);

    return result;
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

tosa::DType translate_client_acc_size(tosa_acc_size_t acc_size)
{
    switch (acc_size)
    {
        case tosa_acc_size_int32_t:
            return tosa::DType::DType_INT32;
        case tosa_acc_size_fp16_t:
            return tosa::DType::DType_FP16;
        case tosa_acc_size_fp32_t:
            return tosa::DType::DType_FP32;
        default:
            return tosa::DType::DType_UNKNOWN;
    }
}

}    // namespace

extern "C"
{

    tosa_status_t tosa_run_argmax(tosa_tensor_t client_input,
                                  const int32_t client_axis,
                                  tosa_tensor_t client_output,
                                  const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        TosaAxisAttribute attr(client_axis);

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_input, "input");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output, "output");

        // Create operator
        auto op =
            new tosa::TosaSerializationOperator(tosa::Op::Op_ARGMAX, tosa::Attribute::Attribute_AxisAttribute, &attr,
                                                getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("argmax", "main", { op }, allTensors(inputTensors, outputTensors),
                                                op->GetInputTensorNames(), op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_avg_pool2d(tosa_tensor_t client_input,
                                      const int32_t client_kernel[2],
                                      const int32_t client_stride[2],
                                      const int32_t client_pad[4],
                                      const tosa_acc_size_t client_acc_size,
                                      const int32_t client_input_zp,
                                      const int32_t client_output_zp,
                                      tosa_tensor_t client_output,
                                      const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        const std::vector<int32_t> pad(&client_pad[0], &client_pad[4]);
        const std::vector<int32_t> kernel(&client_kernel[0], &client_kernel[2]);
        const std::vector<int32_t> stride(&client_stride[0], &client_stride[2]);
        const tosa::DType accum_dtype = translate_client_acc_size(client_acc_size);
        TosaPoolAttribute attr(pad, kernel, stride, client_input_zp, client_output_zp, accum_dtype);

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_input, "input");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output, "output");

        // Create operator
        auto op =
            new tosa::TosaSerializationOperator(tosa::Op::Op_AVG_POOL2D, tosa::Attribute::Attribute_PoolAttribute,
                                                &attr, getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("avg_pool2d", "main", { op }, allTensors(inputTensors, outputTensors),
                                                op->GetInputTensorNames(), op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

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
                                  const bool client_local_bound,
                                  tosa_tensor_t client_output,
                                  const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        const std::vector<int32_t> pad(&client_pad[0], &client_pad[4]);
        const std::vector<int32_t> stride(&client_stride[0], &client_stride[2]);
        const std::vector<int32_t> dilation(&client_dilation[0], &client_dilation[2]);
        TosaConvAttribute attr(pad, stride, dilation, client_input_zp, client_weight_zp, client_local_bound);

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_input, "input");
        addTensor(inputTensors, client_weight, "weight");
        addTensor(inputTensors, client_bias, "bias");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output, "output");

        // Create operator
        auto op =
            new tosa::TosaSerializationOperator(tosa::Op::Op_CONV2D, tosa::Attribute::Attribute_ConvAttribute, &attr,
                                                getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("conv2d", "main", { op }, allTensors(inputTensors, outputTensors),
                                                op->GetInputTensorNames(), op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

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
                                  const bool client_local_bound,
                                  tosa_tensor_t client_output,
                                  const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        const std::vector<int32_t> pad(&client_pad[0], &client_pad[6]);
        const std::vector<int32_t> stride(&client_stride[0], &client_stride[3]);
        const std::vector<int32_t> dilation(&client_dilation[0], &client_dilation[3]);
        TosaConvAttribute attr(pad, stride, dilation, client_input_zp, client_weight_zp, client_local_bound);

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_input, "input");
        addTensor(inputTensors, client_weight, "weight");
        addTensor(inputTensors, client_bias, "bias");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output, "output");

        // Create operator
        auto op =
            new tosa::TosaSerializationOperator(tosa::Op::Op_CONV3D, tosa::Attribute::Attribute_ConvAttribute, &attr,
                                                getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("conv3d", "main", { op }, allTensors(inputTensors, outputTensors),
                                                op->GetInputTensorNames(), op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

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
                                            const bool client_local_bound,
                                            tosa_tensor_t client_output,
                                            const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        const std::vector<int32_t> pad(&client_pad[0], &client_pad[4]);
        const std::vector<int32_t> stride(&client_stride[0], &client_stride[2]);
        const std::vector<int32_t> dilation(&client_dilation[0], &client_dilation[2]);
        TosaConvAttribute attr(pad, stride, dilation, client_input_zp, client_weight_zp, client_local_bound);

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_input, "input");
        addTensor(inputTensors, client_weight, "weight");
        addTensor(inputTensors, client_bias, "bias");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output, "output");

        // Create operator
        auto op =
            new tosa::TosaSerializationOperator(tosa::Op::Op_DEPTHWISE_CONV2D, tosa::Attribute::Attribute_ConvAttribute,
                                                &attr, getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("depthwise_conv2d", "main", { op },
                                                allTensors(inputTensors, outputTensors), op->GetInputTensorNames(),
                                                op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_fft2d(tosa_tensor_t client_input_real,
                                 tosa_tensor_t client_input_imag,
                                 const bool client_inverse,
                                 tosa_tensor_t client_output_real,
                                 const bool client_local_bound,
                                 tosa_tensor_t client_output_imag,
                                 const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        TosaFFTAttribute attr(client_inverse, client_local_bound);

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_input_real, "input_real");
        addTensor(inputTensors, client_input_imag, "input_imag");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output_real, "output_real");
        addTensor(outputTensors, client_output_imag, "output_imag");

        // Create operator
        auto op =
            new tosa::TosaSerializationOperator(tosa::Op::Op_FFT2D, tosa::Attribute::Attribute_FFTAttribute, &attr,
                                                getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("fft2d", "main", { op }, allTensors(inputTensors, outputTensors),
                                                op->GetInputTensorNames(), op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_fully_connected(tosa_tensor_t client_input,
                                           tosa_tensor_t client_weight,
                                           tosa_tensor_t client_bias,
                                           const int32_t client_input_zp,
                                           const int32_t client_weight_zp,
                                           tosa_tensor_t client_output,
                                           const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        TosaFullyConnectedAttribute attr(client_input_zp, client_weight_zp);

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_input, "input");
        addTensor(inputTensors, client_weight, "weight");
        addTensor(inputTensors, client_bias, "bias");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_FULLY_CONNECTED,
                                                      tosa::Attribute::Attribute_FullyConnectedAttribute, &attr,
                                                      getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("fully_connected", "main", { op },
                                                allTensors(inputTensors, outputTensors), op->GetInputTensorNames(),
                                                op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_matmul(tosa_tensor_t client_a,
                                  tosa_tensor_t client_b,
                                  const int32_t client_a_zp,
                                  const int32_t client_b_zp,
                                  tosa_tensor_t client_output,
                                  const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        TosaMatMulAttribute attr(client_a_zp, client_b_zp);

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_a, "a");
        addTensor(inputTensors, client_b, "b");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output, "output");

        // Create operator
        auto op =
            new tosa::TosaSerializationOperator(tosa::Op::Op_MATMUL, tosa::Attribute::Attribute_MatMulAttribute, &attr,
                                                getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("matmul", "main", { op }, allTensors(inputTensors, outputTensors),
                                                op->GetInputTensorNames(), op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_max_pool2d(tosa_tensor_t client_input,
                                      const int32_t client_kernel[2],
                                      const int32_t client_stride[2],
                                      const int32_t client_pad[4],
                                      const int32_t client_input_zp,
                                      const int32_t client_output_zp,
                                      tosa_tensor_t client_output,
                                      const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        const std::vector<int32_t> pad(&client_pad[0], &client_pad[4]);
        const std::vector<int32_t> kernel(&client_kernel[0], &client_kernel[2]);
        const std::vector<int32_t> stride(&client_stride[0], &client_stride[2]);
        const tosa::DType accum_dtype = tosa::DType::DType_FP32;
        TosaPoolAttribute attr(pad, kernel, stride, client_input_zp, client_output_zp, accum_dtype);

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_input, "input");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output, "output");

        // Create operator
        auto op =
            new tosa::TosaSerializationOperator(tosa::Op::Op_MAX_POOL2D, tosa::Attribute::Attribute_PoolAttribute,
                                                &attr, getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("max_pool2d", "main", { op }, allTensors(inputTensors, outputTensors),
                                                op->GetInputTensorNames(), op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_rfft2d(tosa_tensor_t client_input,
                                  tosa_tensor_t client_output_real,
                                  const bool client_local_bound,
                                  tosa_tensor_t client_output_imag,
                                  const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        TosaRFFTAttribute attr(client_local_bound);

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_input, "input");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output_real, "output_real");
        addTensor(outputTensors, client_output_imag, "output_imag");

        // Create operator
        auto op =
            new tosa::TosaSerializationOperator(tosa::Op::Op_RFFT2D, tosa::Attribute::Attribute_RFFTAttribute, &attr,
                                                getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("rfft2d", "main", { op }, allTensors(inputTensors, outputTensors),
                                                op->GetInputTensorNames(), op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

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
                                            const bool client_local_bound,
                                            tosa_tensor_t client_output,
                                            const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        const std::vector<int32_t> out_pad(&client_out_pad[0], &client_out_pad[4]);
        const std::vector<int32_t> stride(&client_stride[0], &client_stride[2]);
        const std::vector<int32_t> out_shape(&client_out_shape[0], &client_out_shape[4]);
        TosaTransposeConvAttribute attr(out_pad, stride, out_shape, client_input_zp, client_weight_zp,
                                        client_local_bound);

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_input, "input");
        addTensor(inputTensors, client_weight, "weight");
        addTensor(inputTensors, client_bias, "bias");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_TRANSPOSE_CONV2D,
                                                      tosa::Attribute::Attribute_TransposeConvAttribute, &attr,
                                                      getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("transpose_conv2d", "main", { op },
                                                allTensors(inputTensors, outputTensors), op->GetInputTensorNames(),
                                                op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_clamp(tosa_tensor_t client_input,
                                 const int32_t client_min_int,
                                 const int32_t client_max_int,
                                 const float client_min_fp,
                                 const float client_max_fp,
                                 tosa_tensor_t client_output,
                                 const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        TosaClampAttribute attr(client_min_int, client_max_int, client_min_fp, client_max_fp);

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_input, "input");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output, "output");

        // Create operator
        auto op =
            new tosa::TosaSerializationOperator(tosa::Op::Op_CLAMP, tosa::Attribute::Attribute_ClampAttribute, &attr,
                                                getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("clamp", "main", { op }, allTensors(inputTensors, outputTensors),
                                                op->GetInputTensorNames(), op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_erf(tosa_tensor_t client_input, tosa_tensor_t client_output, const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_input, "input");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_ERF, tosa::Attribute::Attribute_NONE, &attr,
                                                      getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("erf", "main", { op }, allTensors(inputTensors, outputTensors),
                                                op->GetInputTensorNames(), op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_sigmoid(tosa_tensor_t client_input, tosa_tensor_t client_output, const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_input, "input");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_SIGMOID, tosa::Attribute::Attribute_NONE, &attr,
                                                      getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("sigmoid", "main", { op }, allTensors(inputTensors, outputTensors),
                                                op->GetInputTensorNames(), op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_tanh(tosa_tensor_t client_input, tosa_tensor_t client_output, const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_input, "input");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_TANH, tosa::Attribute::Attribute_NONE, &attr,
                                                      getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("tanh", "main", { op }, allTensors(inputTensors, outputTensors),
                                                op->GetInputTensorNames(), op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_add(tosa_tensor_t client_input1,
                               tosa_tensor_t client_input2,
                               tosa_tensor_t client_output,
                               const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_input1, "input1");
        addTensor(inputTensors, client_input2, "input2");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_ADD, tosa::Attribute::Attribute_NONE, &attr,
                                                      getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("add", "main", { op }, allTensors(inputTensors, outputTensors),
                                                op->GetInputTensorNames(), op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_arithmetic_right_shift(tosa_tensor_t client_input1,
                                                  tosa_tensor_t client_input2,
                                                  const bool client_round,
                                                  tosa_tensor_t client_output,
                                                  const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        TosaArithmeticRightShiftAttribute attr(client_round);

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_input1, "input1");
        addTensor(inputTensors, client_input2, "input2");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_ARITHMETIC_RIGHT_SHIFT,
                                                      tosa::Attribute::Attribute_ArithmeticRightShiftAttribute, &attr,
                                                      getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("arithmetic_right_shift", "main", { op },
                                                allTensors(inputTensors, outputTensors), op->GetInputTensorNames(),
                                                op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_bitwise_and(tosa_tensor_t client_input1,
                                       tosa_tensor_t client_input2,
                                       tosa_tensor_t client_output,
                                       const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_input1, "input1");
        addTensor(inputTensors, client_input2, "input2");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_BITWISE_AND, tosa::Attribute::Attribute_NONE, &attr,
                                                      getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("bitwise_and", "main", { op }, allTensors(inputTensors, outputTensors),
                                                op->GetInputTensorNames(), op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_bitwise_or(tosa_tensor_t client_input1,
                                      tosa_tensor_t client_input2,
                                      tosa_tensor_t client_output,
                                      const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_input1, "input1");
        addTensor(inputTensors, client_input2, "input2");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_BITWISE_OR, tosa::Attribute::Attribute_NONE, &attr,
                                                      getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("bitwise_or", "main", { op }, allTensors(inputTensors, outputTensors),
                                                op->GetInputTensorNames(), op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_bitwise_xor(tosa_tensor_t client_input1,
                                       tosa_tensor_t client_input2,
                                       tosa_tensor_t client_output,
                                       const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_input1, "input1");
        addTensor(inputTensors, client_input2, "input2");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_BITWISE_XOR, tosa::Attribute::Attribute_NONE, &attr,
                                                      getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("bitwise_xor", "main", { op }, allTensors(inputTensors, outputTensors),
                                                op->GetInputTensorNames(), op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_intdiv(tosa_tensor_t client_input1,
                                  tosa_tensor_t client_input2,
                                  tosa_tensor_t client_output,
                                  const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_input1, "input1");
        addTensor(inputTensors, client_input2, "input2");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_INTDIV, tosa::Attribute::Attribute_NONE, &attr,
                                                      getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("intdiv", "main", { op }, allTensors(inputTensors, outputTensors),
                                                op->GetInputTensorNames(), op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_logical_and(tosa_tensor_t client_input1,
                                       tosa_tensor_t client_input2,
                                       tosa_tensor_t client_output,
                                       const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_input1, "input1");
        addTensor(inputTensors, client_input2, "input2");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_LOGICAL_AND, tosa::Attribute::Attribute_NONE, &attr,
                                                      getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("logical_and", "main", { op }, allTensors(inputTensors, outputTensors),
                                                op->GetInputTensorNames(), op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_logical_left_shift(tosa_tensor_t client_input1,
                                              tosa_tensor_t client_input2,
                                              tosa_tensor_t client_output,
                                              const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_input1, "input1");
        addTensor(inputTensors, client_input2, "input2");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output, "output");

        // Create operator
        auto op =
            new tosa::TosaSerializationOperator(tosa::Op::Op_LOGICAL_LEFT_SHIFT, tosa::Attribute::Attribute_NONE, &attr,
                                                getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("logical_left_shift", "main", { op },
                                                allTensors(inputTensors, outputTensors), op->GetInputTensorNames(),
                                                op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_logical_right_shift(tosa_tensor_t client_input1,
                                               tosa_tensor_t client_input2,
                                               tosa_tensor_t client_output,
                                               const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_input1, "input1");
        addTensor(inputTensors, client_input2, "input2");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output, "output");

        // Create operator
        auto op =
            new tosa::TosaSerializationOperator(tosa::Op::Op_LOGICAL_RIGHT_SHIFT, tosa::Attribute::Attribute_NONE,
                                                &attr, getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("logical_right_shift", "main", { op },
                                                allTensors(inputTensors, outputTensors), op->GetInputTensorNames(),
                                                op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_logical_or(tosa_tensor_t client_input1,
                                      tosa_tensor_t client_input2,
                                      tosa_tensor_t client_output,
                                      const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_input1, "input1");
        addTensor(inputTensors, client_input2, "input2");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_LOGICAL_OR, tosa::Attribute::Attribute_NONE, &attr,
                                                      getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("logical_or", "main", { op }, allTensors(inputTensors, outputTensors),
                                                op->GetInputTensorNames(), op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_logical_xor(tosa_tensor_t client_input1,
                                       tosa_tensor_t client_input2,
                                       tosa_tensor_t client_output,
                                       const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_input1, "input1");
        addTensor(inputTensors, client_input2, "input2");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_LOGICAL_XOR, tosa::Attribute::Attribute_NONE, &attr,
                                                      getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("logical_xor", "main", { op }, allTensors(inputTensors, outputTensors),
                                                op->GetInputTensorNames(), op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_maximum(tosa_tensor_t client_input1,
                                   tosa_tensor_t client_input2,
                                   tosa_tensor_t client_output,
                                   const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_input1, "input1");
        addTensor(inputTensors, client_input2, "input2");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_MAXIMUM, tosa::Attribute::Attribute_NONE, &attr,
                                                      getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("maximum", "main", { op }, allTensors(inputTensors, outputTensors),
                                                op->GetInputTensorNames(), op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_minimum(tosa_tensor_t client_input1,
                                   tosa_tensor_t client_input2,
                                   tosa_tensor_t client_output,
                                   const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_input1, "input1");
        addTensor(inputTensors, client_input2, "input2");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_MINIMUM, tosa::Attribute::Attribute_NONE, &attr,
                                                      getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("minimum", "main", { op }, allTensors(inputTensors, outputTensors),
                                                op->GetInputTensorNames(), op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_mul(tosa_tensor_t client_input1,
                               tosa_tensor_t client_input2,
                               const int32_t client_shift,
                               tosa_tensor_t client_output,
                               const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        TosaMulAttribute attr(client_shift);

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_input1, "input1");
        addTensor(inputTensors, client_input2, "input2");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_MUL, tosa::Attribute::Attribute_MulAttribute, &attr,
                                                      getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("mul", "main", { op }, allTensors(inputTensors, outputTensors),
                                                op->GetInputTensorNames(), op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_pow(tosa_tensor_t client_input1,
                               tosa_tensor_t client_input2,
                               tosa_tensor_t client_output,
                               const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_input1, "input1");
        addTensor(inputTensors, client_input2, "input2");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_POW, tosa::Attribute::Attribute_NONE, &attr,
                                                      getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("pow", "main", { op }, allTensors(inputTensors, outputTensors),
                                                op->GetInputTensorNames(), op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_sub(tosa_tensor_t client_input1,
                               tosa_tensor_t client_input2,
                               tosa_tensor_t client_output,
                               const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_input1, "input1");
        addTensor(inputTensors, client_input2, "input2");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_SUB, tosa::Attribute::Attribute_NONE, &attr,
                                                      getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("sub", "main", { op }, allTensors(inputTensors, outputTensors),
                                                op->GetInputTensorNames(), op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_table(tosa_tensor_t client_input,
                                 const int32_t client_table_len,
                                 const int16_t client_table[],
                                 tosa_tensor_t client_output,
                                 const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        const std::vector<int16_t> table(&client_table[0], &client_table[0] + client_table_len);
        TosaTableAttribute attr(table);

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_input, "input");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output, "output");

        // Create operator
        auto op =
            new tosa::TosaSerializationOperator(tosa::Op::Op_TABLE, tosa::Attribute::Attribute_TableAttribute, &attr,
                                                getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("table", "main", { op }, allTensors(inputTensors, outputTensors),
                                                op->GetInputTensorNames(), op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_abs(tosa_tensor_t client_input1, tosa_tensor_t client_output, const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_input1, "input1");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_ABS, tosa::Attribute::Attribute_NONE, &attr,
                                                      getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("abs", "main", { op }, allTensors(inputTensors, outputTensors),
                                                op->GetInputTensorNames(), op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

        return tosa_status_valid;
    }

    tosa_status_t
        tosa_run_bitwise_not(tosa_tensor_t client_input1, tosa_tensor_t client_output, const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_input1, "input1");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_BITWISE_NOT, tosa::Attribute::Attribute_NONE, &attr,
                                                      getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("bitwise_not", "main", { op }, allTensors(inputTensors, outputTensors),
                                                op->GetInputTensorNames(), op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_ceil(tosa_tensor_t client_input1, tosa_tensor_t client_output, const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_input1, "input1");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_CEIL, tosa::Attribute::Attribute_NONE, &attr,
                                                      getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("ceil", "main", { op }, allTensors(inputTensors, outputTensors),
                                                op->GetInputTensorNames(), op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_clz(tosa_tensor_t client_input1, tosa_tensor_t client_output, const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_input1, "input1");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_CLZ, tosa::Attribute::Attribute_NONE, &attr,
                                                      getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("clz", "main", { op }, allTensors(inputTensors, outputTensors),
                                                op->GetInputTensorNames(), op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_exp(tosa_tensor_t client_input1, tosa_tensor_t client_output, const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_input1, "input1");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_EXP, tosa::Attribute::Attribute_NONE, &attr,
                                                      getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("exp", "main", { op }, allTensors(inputTensors, outputTensors),
                                                op->GetInputTensorNames(), op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_floor(tosa_tensor_t client_input1, tosa_tensor_t client_output, const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_input1, "input1");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_FLOOR, tosa::Attribute::Attribute_NONE, &attr,
                                                      getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("floor", "main", { op }, allTensors(inputTensors, outputTensors),
                                                op->GetInputTensorNames(), op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_log(tosa_tensor_t client_input1, tosa_tensor_t client_output, const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_input1, "input1");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_LOG, tosa::Attribute::Attribute_NONE, &attr,
                                                      getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("log", "main", { op }, allTensors(inputTensors, outputTensors),
                                                op->GetInputTensorNames(), op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

        return tosa_status_valid;
    }

    tosa_status_t
        tosa_run_logical_not(tosa_tensor_t client_input1, tosa_tensor_t client_output, const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_input1, "input1");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_LOGICAL_NOT, tosa::Attribute::Attribute_NONE, &attr,
                                                      getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("logical_not", "main", { op }, allTensors(inputTensors, outputTensors),
                                                op->GetInputTensorNames(), op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_negate(tosa_tensor_t client_input1,
                                  const int32_t client_input1_zp,
                                  const int32_t client_output_zp,
                                  tosa_tensor_t client_output,
                                  const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        TosaNegateAttribute attr(client_input1_zp, client_output_zp);

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_input1, "input1");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output, "output");

        // Create operator
        auto op =
            new tosa::TosaSerializationOperator(tosa::Op::Op_NEGATE, tosa::Attribute::Attribute_NegateAttribute, &attr,
                                                getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("negate", "main", { op }, allTensors(inputTensors, outputTensors),
                                                op->GetInputTensorNames(), op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

        return tosa_status_valid;
    }

    tosa_status_t
        tosa_run_reciprocal(tosa_tensor_t client_input1, tosa_tensor_t client_output, const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_input1, "input1");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_RECIPROCAL, tosa::Attribute::Attribute_NONE, &attr,
                                                      getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("reciprocal", "main", { op }, allTensors(inputTensors, outputTensors),
                                                op->GetInputTensorNames(), op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_rsqrt(tosa_tensor_t client_input1, tosa_tensor_t client_output, const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_input1, "input1");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_RSQRT, tosa::Attribute::Attribute_NONE, &attr,
                                                      getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("rsqrt", "main", { op }, allTensors(inputTensors, outputTensors),
                                                op->GetInputTensorNames(), op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_select(tosa_tensor_t client_input1,
                                  tosa_tensor_t client_input2,
                                  tosa_tensor_t client_input3,
                                  tosa_tensor_t client_output,
                                  const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_input1, "input1");
        addTensor(inputTensors, client_input2, "input2");
        addTensor(inputTensors, client_input3, "input3");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_SELECT, tosa::Attribute::Attribute_NONE, &attr,
                                                      getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("select", "main", { op }, allTensors(inputTensors, outputTensors),
                                                op->GetInputTensorNames(), op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_equal(tosa_tensor_t client_input1,
                                 tosa_tensor_t client_input2,
                                 tosa_tensor_t client_output,
                                 const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_input1, "input1");
        addTensor(inputTensors, client_input2, "input2");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_EQUAL, tosa::Attribute::Attribute_NONE, &attr,
                                                      getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("equal", "main", { op }, allTensors(inputTensors, outputTensors),
                                                op->GetInputTensorNames(), op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_greater(tosa_tensor_t client_input1,
                                   tosa_tensor_t client_input2,
                                   tosa_tensor_t client_output,
                                   const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_input1, "input1");
        addTensor(inputTensors, client_input2, "input2");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_GREATER, tosa::Attribute::Attribute_NONE, &attr,
                                                      getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("greater", "main", { op }, allTensors(inputTensors, outputTensors),
                                                op->GetInputTensorNames(), op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_greater_equal(tosa_tensor_t client_input1,
                                         tosa_tensor_t client_input2,
                                         tosa_tensor_t client_output,
                                         const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_input1, "input1");
        addTensor(inputTensors, client_input2, "input2");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output, "output");

        // Create operator
        auto op =
            new tosa::TosaSerializationOperator(tosa::Op::Op_GREATER_EQUAL, tosa::Attribute::Attribute_NONE, &attr,
                                                getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("greater_equal", "main", { op },
                                                allTensors(inputTensors, outputTensors), op->GetInputTensorNames(),
                                                op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_reduce_all(tosa_tensor_t client_input,
                                      const int32_t client_axis,
                                      tosa_tensor_t client_output,
                                      const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        TosaAxisAttribute attr(client_axis);

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_input, "input");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output, "output");

        // Create operator
        auto op =
            new tosa::TosaSerializationOperator(tosa::Op::Op_REDUCE_ALL, tosa::Attribute::Attribute_AxisAttribute,
                                                &attr, getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("reduce_all", "main", { op }, allTensors(inputTensors, outputTensors),
                                                op->GetInputTensorNames(), op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_reduce_any(tosa_tensor_t client_input,
                                      const int32_t client_axis,
                                      tosa_tensor_t client_output,
                                      const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        TosaAxisAttribute attr(client_axis);

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_input, "input");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output, "output");

        // Create operator
        auto op =
            new tosa::TosaSerializationOperator(tosa::Op::Op_REDUCE_ANY, tosa::Attribute::Attribute_AxisAttribute,
                                                &attr, getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("reduce_any", "main", { op }, allTensors(inputTensors, outputTensors),
                                                op->GetInputTensorNames(), op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_reduce_max(tosa_tensor_t client_input,
                                      const int32_t client_axis,
                                      tosa_tensor_t client_output,
                                      const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        TosaAxisAttribute attr(client_axis);

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_input, "input");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output, "output");

        // Create operator
        auto op =
            new tosa::TosaSerializationOperator(tosa::Op::Op_REDUCE_MAX, tosa::Attribute::Attribute_AxisAttribute,
                                                &attr, getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("reduce_max", "main", { op }, allTensors(inputTensors, outputTensors),
                                                op->GetInputTensorNames(), op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_reduce_min(tosa_tensor_t client_input,
                                      const int32_t client_axis,
                                      tosa_tensor_t client_output,
                                      const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        TosaAxisAttribute attr(client_axis);

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_input, "input");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output, "output");

        // Create operator
        auto op =
            new tosa::TosaSerializationOperator(tosa::Op::Op_REDUCE_MIN, tosa::Attribute::Attribute_AxisAttribute,
                                                &attr, getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("reduce_min", "main", { op }, allTensors(inputTensors, outputTensors),
                                                op->GetInputTensorNames(), op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_reduce_product(tosa_tensor_t client_input,
                                          const int32_t client_axis,
                                          tosa_tensor_t client_output,
                                          const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        TosaAxisAttribute attr(client_axis);

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_input, "input");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output, "output");

        // Create operator
        auto op =
            new tosa::TosaSerializationOperator(tosa::Op::Op_REDUCE_PRODUCT, tosa::Attribute::Attribute_AxisAttribute,
                                                &attr, getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("reduce_product", "main", { op },
                                                allTensors(inputTensors, outputTensors), op->GetInputTensorNames(),
                                                op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_reduce_sum(tosa_tensor_t client_input,
                                      const int32_t client_axis,
                                      tosa_tensor_t client_output,
                                      const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        TosaAxisAttribute attr(client_axis);

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_input, "input");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output, "output");

        // Create operator
        auto op =
            new tosa::TosaSerializationOperator(tosa::Op::Op_REDUCE_SUM, tosa::Attribute::Attribute_AxisAttribute,
                                                &attr, getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("reduce_sum", "main", { op }, allTensors(inputTensors, outputTensors),
                                                op->GetInputTensorNames(), op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_concat(const tosa_tensor_list_t client_input1,
                                  const int32_t client_axis,
                                  tosa_tensor_t client_output,
                                  const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        TosaAxisAttribute attr(client_axis);

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        for (int i = 0; i < client_input1.size; i++)
        {
            addTensor(inputTensors, client_input1.tensors[i], "input1-" + std::to_string(i));
        }

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output, "output");

        // Create operator
        auto op =
            new tosa::TosaSerializationOperator(tosa::Op::Op_CONCAT, tosa::Attribute::Attribute_AxisAttribute, &attr,
                                                getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("concat", "main", { op }, allTensors(inputTensors, outputTensors),
                                                op->GetInputTensorNames(), op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_pad(tosa_tensor_t client_input1,
                               tosa_tensor_t client_padding,
                               const int32_t client_pad_const_int,
                               const float client_pad_const_fp,
                               tosa_tensor_t client_output,
                               const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        std::vector<int32_t> padding;
        size_t padding_size   = client_padding.size / sizeof(int32_t);
        int32_t* padding_data = reinterpret_cast<int32_t*>(client_padding.data);
        padding.assign(padding_data, padding_data + padding_size);
        TosaPadAttribute attr(padding, client_pad_const_int, client_pad_const_fp);

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_input1, "input1");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_PAD, tosa::Attribute::Attribute_PadAttribute, &attr,
                                                      getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("pad", "main", { op }, allTensors(inputTensors, outputTensors),
                                                op->GetInputTensorNames(), op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_dim(tosa_tensor_t client_input1,
                               const int32_t client_axis,
                               tosa_tensor_t client_output,
                               const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        TosaAxisAttribute attr(client_axis);

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_input1, "input1");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_DIM, tosa::Attribute::Attribute_AxisAttribute, &attr,
                                                      getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("dim", "main", { op }, allTensors(inputTensors, outputTensors),
                                                op->GetInputTensorNames(), op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_reshape(tosa_tensor_t client_input1,
                                   tosa_tensor_t client_shape,
                                   tosa_tensor_t client_output,
                                   const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        std::vector<int32_t> shape;
        size_t shape_size   = client_shape.size / sizeof(int32_t);
        int32_t* shape_data = reinterpret_cast<int32_t*>(client_shape.data);
        shape.assign(shape_data, shape_data + shape_size);
        TosaReshapeAttribute attr(shape);

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_input1, "input1");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output, "output");

        // Create operator
        auto op =
            new tosa::TosaSerializationOperator(tosa::Op::Op_RESHAPE, tosa::Attribute::Attribute_ReshapeAttribute,
                                                &attr, getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("reshape", "main", { op }, allTensors(inputTensors, outputTensors),
                                                op->GetInputTensorNames(), op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_reverse(tosa_tensor_t client_input,
                                   const int32_t client_axis,
                                   tosa_tensor_t client_output,
                                   const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        TosaAxisAttribute attr(client_axis);

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_input, "input");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output, "output");

        // Create operator
        auto op =
            new tosa::TosaSerializationOperator(tosa::Op::Op_REVERSE, tosa::Attribute::Attribute_AxisAttribute, &attr,
                                                getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("reverse", "main", { op }, allTensors(inputTensors, outputTensors),
                                                op->GetInputTensorNames(), op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_slice(tosa_tensor_t client_input1,
                                 const int32_t client_start_len,
                                 const int32_t client_start[],
                                 const int32_t client_size_len,
                                 const int32_t client_size[],
                                 tosa_tensor_t client_output,
                                 const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        const std::vector<int32_t> start(&client_start[0], &client_start[0] + client_start_len);
        const std::vector<int32_t> size(&client_size[0], &client_size[0] + client_size_len);
        TosaSliceAttribute attr(start, size);

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_input1, "input1");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output, "output");

        // Create operator
        auto op =
            new tosa::TosaSerializationOperator(tosa::Op::Op_SLICE, tosa::Attribute::Attribute_SliceAttribute, &attr,
                                                getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("slice", "main", { op }, allTensors(inputTensors, outputTensors),
                                                op->GetInputTensorNames(), op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_tile(tosa_tensor_t client_input1,
                                tosa_tensor_t client_multiples,
                                tosa_tensor_t client_output,
                                const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        std::vector<int32_t> multiples;
        size_t multiples_size   = client_multiples.size / sizeof(int32_t);
        int32_t* multiples_data = reinterpret_cast<int32_t*>(client_multiples.data);
        multiples.assign(multiples_data, multiples_data + multiples_size);
        TosaTileAttribute attr(multiples);

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_input1, "input1");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output, "output");

        // Create operator
        auto op =
            new tosa::TosaSerializationOperator(tosa::Op::Op_TILE, tosa::Attribute::Attribute_TileAttribute, &attr,
                                                getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("tile", "main", { op }, allTensors(inputTensors, outputTensors),
                                                op->GetInputTensorNames(), op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_transpose(tosa_tensor_t client_input1,
                                     const int32_t client_perms_len,
                                     const int32_t client_perms[],
                                     tosa_tensor_t client_output,
                                     const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        const std::vector<int32_t> perms(&client_perms[0], &client_perms[0] + client_perms_len);
        TosaTransposeAttribute attr(perms);

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_input1, "input1");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output, "output");

        // Create operator
        auto op =
            new tosa::TosaSerializationOperator(tosa::Op::Op_TRANSPOSE, tosa::Attribute::Attribute_TransposeAttribute,
                                                &attr, getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("transpose", "main", { op }, allTensors(inputTensors, outputTensors),
                                                op->GetInputTensorNames(), op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_gather(tosa_tensor_t client_values,
                                  tosa_tensor_t client_indices,
                                  tosa_tensor_t client_output,
                                  const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_values, "values");
        addTensor(inputTensors, client_indices, "indices");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_GATHER, tosa::Attribute::Attribute_NONE, &attr,
                                                      getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("gather", "main", { op }, allTensors(inputTensors, outputTensors),
                                                op->GetInputTensorNames(), op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_scatter(tosa_tensor_t client_values_in,
                                   tosa_tensor_t client_indices,
                                   tosa_tensor_t client_input,
                                   tosa_tensor_t client_values_out,
                                   const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_values_in, "values_in");
        addTensor(inputTensors, client_indices, "indices");
        addTensor(inputTensors, client_input, "input");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_values_out, "values_out");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_SCATTER, tosa::Attribute::Attribute_NONE, &attr,
                                                      getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("scatter", "main", { op }, allTensors(inputTensors, outputTensors),
                                                op->GetInputTensorNames(), op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_resize(tosa_tensor_t client_input,
                                  tosa_tensor_t client_scale,
                                  tosa_tensor_t client_offset,
                                  tosa_tensor_t client_border,
                                  const tosa_mode_t client_mode,
                                  tosa_tensor_t client_output,
                                  const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        std::vector<int16_t> scale;
        size_t scale_size   = client_scale.size / sizeof(int16_t);
        int16_t* scale_data = reinterpret_cast<int16_t*>(client_scale.data);
        scale.assign(scale_data, scale_data + scale_size);
        std::vector<int16_t> offset;
        size_t offset_size   = client_offset.size / sizeof(int16_t);
        int16_t* offset_data = reinterpret_cast<int16_t*>(client_offset.data);
        offset.assign(offset_data, offset_data + offset_size);
        std::vector<int16_t> border;
        size_t border_size   = client_border.size / sizeof(int16_t);
        int16_t* border_data = reinterpret_cast<int16_t*>(client_border.data);
        border.assign(border_data, border_data + border_size);
        const ResizeMode mode = translate_client_tosa_mode(client_mode);
        TosaResizeAttribute attr(scale, offset, border, mode);

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_input, "input");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output, "output");

        // Create operator
        auto op =
            new tosa::TosaSerializationOperator(tosa::Op::Op_RESIZE, tosa::Attribute::Attribute_ResizeAttribute, &attr,
                                                getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("resize", "main", { op }, allTensors(inputTensors, outputTensors),
                                                op->GetInputTensorNames(), op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_cast(tosa_tensor_t client_input, tosa_tensor_t client_output, const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_input, "input");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_CAST, tosa::Attribute::Attribute_NONE, &attr,
                                                      getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("cast", "main", { op }, allTensors(inputTensors, outputTensors),
                                                op->GetInputTensorNames(), op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

        return tosa_status_valid;
    }

    tosa_status_t tosa_run_rescale(tosa_tensor_t client_input,
                                   tosa_tensor_t client_output,
                                   const int32_t client_input_zp,
                                   const int32_t client_output_zp,
                                   const int32_t client_multiplier_len,
                                   const int32_t client_multiplier[],
                                   const int32_t client_shift_len,
                                   const int32_t client_shift[],
                                   const bool client_scale32,
                                   const bool client_double_round,
                                   const bool client_input_unsigned,
                                   const bool client_output_unsigned,
                                   const bool client_per_channel,
                                   const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        const std::vector<int32_t> multiplier(&client_multiplier[0], &client_multiplier[0] + client_multiplier_len);
        const std::vector<int32_t> shift(&client_shift[0], &client_shift[0] + client_shift_len);
        TosaRescaleAttribute attr(client_input_zp, client_output_zp, multiplier, shift, client_scale32,
                                  client_double_round, client_per_channel, client_input_unsigned,
                                  client_output_unsigned);

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_input, "input");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output, "output");

        // Create operator
        auto op =
            new tosa::TosaSerializationOperator(tosa::Op::Op_RESCALE, tosa::Attribute::Attribute_RescaleAttribute,
                                                &attr, getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("rescale", "main", { op }, allTensors(inputTensors, outputTensors),
                                                op->GetInputTensorNames(), op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

        return tosa_status_valid;
    }

    tosa_status_t
        tosa_run_identity(tosa_tensor_t client_input1, tosa_tensor_t client_output, const func_ctx_t& func_ctx)
    {
        // Create operator attributes
        TosaNoneAttribute attr;

        // Create tensors
        std::vector<TosaTensorInfo> inputTensors;
        addTensor(inputTensors, client_input1, "input1");

        std::vector<TosaTensorInfo> outputTensors;
        addTensor(outputTensors, client_output, "output");

        // Create operator
        auto op = new tosa::TosaSerializationOperator(tosa::Op::Op_IDENTITY, tosa::Attribute::Attribute_NONE, &attr,
                                                      getTensorNames(inputTensors), getTensorNames(outputTensors));

        // Create a tosa single-op basic block
        tosa::TosaSerializationBasicBlock block("identity", "main", { op }, allTensors(inputTensors, outputTensors),
                                                op->GetInputTensorNames(), op->GetOutputTensorNames());

        // Setup model
        TosaReference::ModelRunnerImpl runner(func_ctx.func_config, func_ctx.func_debug);
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.initialize(block));

        TOSA_RETURN_ON_ERROR(setInputTensors(runner, inputTensors));

        // Execute
        TOSA_RETURN_ON_GRAPH_STATUS_ERROR(runner.run());

        // Extract outputs
        TOSA_RETURN_ON_ERROR(getOutputTensors(runner, outputTensors));

        return tosa_status_valid;
    }

}    // extern "C"
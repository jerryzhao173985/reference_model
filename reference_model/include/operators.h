
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

#ifndef OPERATORS_H_
#define OPERATORS_H_

#include "func_config.h"
#include "func_debug.h"
#include "types.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */

    struct func_ctx_t
    {
        func_config_t func_config = func_config_t{};
        func_debug_t func_debug   = func_debug_t{};
    };

    tosa_status_t tosa_run_argmax(tosa_tensor_t client_input,
                                  const int32_t client_axis,
                                  tosa_tensor_t client_output,
                                  const func_ctx_t& func_ctx);

    tosa_status_t tosa_run_avg_pool2d(tosa_tensor_t client_input,
                                      const int32_t client_kernel[2],
                                      const int32_t client_stride[2],
                                      const int32_t client_pad[4],
                                      const tosa_acc_size_t client_acc_size,
                                      const int32_t client_input_zp,
                                      const int32_t client_output_zp,
                                      tosa_tensor_t client_output,
                                      const func_ctx_t& func_ctx);

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
                                  const func_ctx_t& func_ctx);

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
                                  const func_ctx_t& func_ctx);

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
                                            const func_ctx_t& func_ctx);

    tosa_status_t tosa_run_fft2d(tosa_tensor_t client_input_real,
                                 tosa_tensor_t client_input_imag,
                                 const bool client_inverse,
                                 tosa_tensor_t client_output_real,
                                 const bool client_local_bound,
                                 tosa_tensor_t client_output_imag,
                                 const func_ctx_t& func_ctx);

    tosa_status_t tosa_run_fully_connected(tosa_tensor_t client_input,
                                           tosa_tensor_t client_weight,
                                           tosa_tensor_t client_bias,
                                           const int32_t client_input_zp,
                                           const int32_t client_weight_zp,
                                           tosa_tensor_t client_output,
                                           const func_ctx_t& func_ctx);

    tosa_status_t tosa_run_matmul(tosa_tensor_t client_a,
                                  tosa_tensor_t client_b,
                                  const int32_t client_a_zp,
                                  const int32_t client_b_zp,
                                  tosa_tensor_t client_output,
                                  const func_ctx_t& func_ctx);

    tosa_status_t tosa_run_max_pool2d(tosa_tensor_t client_input,
                                      const int32_t client_kernel[2],
                                      const int32_t client_stride[2],
                                      const int32_t client_pad[4],
                                      const int32_t client_input_zp,
                                      const int32_t client_output_zp,
                                      tosa_tensor_t client_output,
                                      const func_ctx_t& func_ctx);

    tosa_status_t tosa_run_rfft2d(tosa_tensor_t client_input,
                                  tosa_tensor_t client_output_real,
                                  const bool client_local_bound,
                                  tosa_tensor_t client_output_imag,
                                  const func_ctx_t& func_ctx);

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
                                            const func_ctx_t& func_ctx);

    tosa_status_t tosa_run_clamp(tosa_tensor_t client_input,
                                 const int32_t client_min_int,
                                 const int32_t client_max_int,
                                 const float client_min_fp,
                                 const float client_max_fp,
                                 tosa_tensor_t client_output,
                                 const func_ctx_t& func_ctx);

    tosa_status_t tosa_run_erf(tosa_tensor_t client_input, tosa_tensor_t client_output, const func_ctx_t& func_ctx);

    tosa_status_t tosa_run_sigmoid(tosa_tensor_t client_input, tosa_tensor_t client_output, const func_ctx_t& func_ctx);

    tosa_status_t tosa_run_tanh(tosa_tensor_t client_input, tosa_tensor_t client_output, const func_ctx_t& func_ctx);

    tosa_status_t tosa_run_add(tosa_tensor_t client_input1,
                               tosa_tensor_t client_input2,
                               tosa_tensor_t client_output,
                               const func_ctx_t& func_ctx);

    tosa_status_t tosa_run_arithmetic_right_shift(tosa_tensor_t client_input1,
                                                  tosa_tensor_t client_input2,
                                                  const bool client_round,
                                                  tosa_tensor_t client_output,
                                                  const func_ctx_t& func_ctx);

    tosa_status_t tosa_run_bitwise_and(tosa_tensor_t client_input1,
                                       tosa_tensor_t client_input2,
                                       tosa_tensor_t client_output,
                                       const func_ctx_t& func_ctx);

    tosa_status_t tosa_run_bitwise_or(tosa_tensor_t client_input1,
                                      tosa_tensor_t client_input2,
                                      tosa_tensor_t client_output,
                                      const func_ctx_t& func_ctx);

    tosa_status_t tosa_run_bitwise_xor(tosa_tensor_t client_input1,
                                       tosa_tensor_t client_input2,
                                       tosa_tensor_t client_output,
                                       const func_ctx_t& func_ctx);

    tosa_status_t tosa_run_intdiv(tosa_tensor_t client_input1,
                                  tosa_tensor_t client_input2,
                                  tosa_tensor_t client_output,
                                  const func_ctx_t& func_ctx);

    tosa_status_t tosa_run_logical_and(tosa_tensor_t client_input1,
                                       tosa_tensor_t client_input2,
                                       tosa_tensor_t client_output,
                                       const func_ctx_t& func_ctx);

    tosa_status_t tosa_run_logical_left_shift(tosa_tensor_t client_input1,
                                              tosa_tensor_t client_input2,
                                              tosa_tensor_t client_output,
                                              const func_ctx_t& func_ctx);

    tosa_status_t tosa_run_logical_right_shift(tosa_tensor_t client_input1,
                                               tosa_tensor_t client_input2,
                                               tosa_tensor_t client_output,
                                               const func_ctx_t& func_ctx);

    tosa_status_t tosa_run_logical_or(tosa_tensor_t client_input1,
                                      tosa_tensor_t client_input2,
                                      tosa_tensor_t client_output,
                                      const func_ctx_t& func_ctx);

    tosa_status_t tosa_run_logical_xor(tosa_tensor_t client_input1,
                                       tosa_tensor_t client_input2,
                                       tosa_tensor_t client_output,
                                       const func_ctx_t& func_ctx);

    tosa_status_t tosa_run_maximum(tosa_tensor_t client_input1,
                                   tosa_tensor_t client_input2,
                                   tosa_tensor_t client_output,
                                   const func_ctx_t& func_ctx);

    tosa_status_t tosa_run_minimum(tosa_tensor_t client_input1,
                                   tosa_tensor_t client_input2,
                                   tosa_tensor_t client_output,
                                   const func_ctx_t& func_ctx);

    tosa_status_t tosa_run_mul(tosa_tensor_t client_input1,
                               tosa_tensor_t client_input2,
                               const int32_t client_shift,
                               tosa_tensor_t client_output,
                               const func_ctx_t& func_ctx);

    tosa_status_t tosa_run_pow(tosa_tensor_t client_input1,
                               tosa_tensor_t client_input2,
                               tosa_tensor_t client_output,
                               const func_ctx_t& func_ctx);

    tosa_status_t tosa_run_sub(tosa_tensor_t client_input1,
                               tosa_tensor_t client_input2,
                               tosa_tensor_t client_output,
                               const func_ctx_t& func_ctx);

    tosa_status_t tosa_run_table(tosa_tensor_t client_input,
                                 const int32_t client_table_len,
                                 const int16_t client_table[],
                                 tosa_tensor_t client_output,
                                 const func_ctx_t& func_ctx);

    tosa_status_t tosa_run_abs(tosa_tensor_t client_input1, tosa_tensor_t client_output, const func_ctx_t& func_ctx);

    tosa_status_t
        tosa_run_bitwise_not(tosa_tensor_t client_input1, tosa_tensor_t client_output, const func_ctx_t& func_ctx);

    tosa_status_t tosa_run_ceil(tosa_tensor_t client_input1, tosa_tensor_t client_output, const func_ctx_t& func_ctx);

    tosa_status_t tosa_run_clz(tosa_tensor_t client_input1, tosa_tensor_t client_output, const func_ctx_t& func_ctx);

    tosa_status_t tosa_run_exp(tosa_tensor_t client_input1, tosa_tensor_t client_output, const func_ctx_t& func_ctx);

    tosa_status_t tosa_run_floor(tosa_tensor_t client_input1, tosa_tensor_t client_output, const func_ctx_t& func_ctx);

    tosa_status_t tosa_run_log(tosa_tensor_t client_input1, tosa_tensor_t client_output, const func_ctx_t& func_ctx);

    tosa_status_t
        tosa_run_logical_not(tosa_tensor_t client_input1, tosa_tensor_t client_output, const func_ctx_t& func_ctx);

    tosa_status_t tosa_run_negate(tosa_tensor_t client_input1,
                                  const int32_t client_input1_zp,
                                  const int32_t client_output_zp,
                                  tosa_tensor_t client_output,
                                  const func_ctx_t& func_ctx);

    tosa_status_t
        tosa_run_reciprocal(tosa_tensor_t client_input1, tosa_tensor_t client_output, const func_ctx_t& func_ctx);

    tosa_status_t tosa_run_rsqrt(tosa_tensor_t client_input1, tosa_tensor_t client_output, const func_ctx_t& func_ctx);

    tosa_status_t tosa_run_select(tosa_tensor_t client_input1,
                                  tosa_tensor_t client_input2,
                                  tosa_tensor_t client_input3,
                                  tosa_tensor_t client_output,
                                  const func_ctx_t& func_ctx);

    tosa_status_t tosa_run_equal(tosa_tensor_t client_input1,
                                 tosa_tensor_t client_input2,
                                 tosa_tensor_t client_output,
                                 const func_ctx_t& func_ctx);

    tosa_status_t tosa_run_greater(tosa_tensor_t client_input1,
                                   tosa_tensor_t client_input2,
                                   tosa_tensor_t client_output,
                                   const func_ctx_t& func_ctx);

    tosa_status_t tosa_run_greater_equal(tosa_tensor_t client_input1,
                                         tosa_tensor_t client_input2,
                                         tosa_tensor_t client_output,
                                         const func_ctx_t& func_ctx);

    tosa_status_t tosa_run_reduce_all(tosa_tensor_t client_input,
                                      const int32_t client_axis,
                                      tosa_tensor_t client_output,
                                      const func_ctx_t& func_ctx);

    tosa_status_t tosa_run_reduce_any(tosa_tensor_t client_input,
                                      const int32_t client_axis,
                                      tosa_tensor_t client_output,
                                      const func_ctx_t& func_ctx);

    tosa_status_t tosa_run_reduce_max(tosa_tensor_t client_input,
                                      const int32_t client_axis,
                                      tosa_tensor_t client_output,
                                      const func_ctx_t& func_ctx);

    tosa_status_t tosa_run_reduce_min(tosa_tensor_t client_input,
                                      const int32_t client_axis,
                                      tosa_tensor_t client_output,
                                      const func_ctx_t& func_ctx);

    tosa_status_t tosa_run_reduce_product(tosa_tensor_t client_input,
                                          const int32_t client_axis,
                                          tosa_tensor_t client_output,
                                          const func_ctx_t& func_ctx);

    tosa_status_t tosa_run_reduce_sum(tosa_tensor_t client_input,
                                      const int32_t client_axis,
                                      tosa_tensor_t client_output,
                                      const func_ctx_t& func_ctx);

    tosa_status_t tosa_run_concat(const tosa_tensor_list_t client_input1,
                                  const int32_t client_axis,
                                  tosa_tensor_t client_output,
                                  const func_ctx_t& func_ctx);

    tosa_status_t tosa_run_pad(tosa_tensor_t client_input1,
                               tosa_tensor_t client_padding,
                               const int32_t client_pad_const_int,
                               const float client_pad_const_fp,
                               tosa_tensor_t client_output,
                               const func_ctx_t& func_ctx);

    tosa_status_t tosa_run_dim(tosa_tensor_t client_input1,
                               const int32_t client_axis,
                               tosa_tensor_t client_output,
                               const func_ctx_t& func_ctx);

    tosa_status_t tosa_run_reshape(tosa_tensor_t client_input1,
                                   tosa_tensor_t client_shape,
                                   tosa_tensor_t client_output,
                                   const func_ctx_t& func_ctx);

    tosa_status_t tosa_run_reverse(tosa_tensor_t client_input,
                                   const int32_t client_axis,
                                   tosa_tensor_t client_output,
                                   const func_ctx_t& func_ctx);

    tosa_status_t tosa_run_slice(tosa_tensor_t client_input1,
                                 const int32_t client_start_len,
                                 const int32_t client_start[],
                                 const int32_t client_size_len,
                                 const int32_t client_size[],
                                 tosa_tensor_t client_output,
                                 const func_ctx_t& func_ctx);

    tosa_status_t tosa_run_tile(tosa_tensor_t client_input1,
                                tosa_tensor_t client_multiples,
                                tosa_tensor_t client_output,
                                const func_ctx_t& func_ctx);

    tosa_status_t tosa_run_transpose(tosa_tensor_t client_input1,
                                     const int32_t client_perms_len,
                                     const int32_t client_perms[],
                                     tosa_tensor_t client_output,
                                     const func_ctx_t& func_ctx);

    tosa_status_t tosa_run_gather(tosa_tensor_t client_values,
                                  tosa_tensor_t client_indices,
                                  tosa_tensor_t client_output,
                                  const func_ctx_t& func_ctx);

    tosa_status_t tosa_run_scatter(tosa_tensor_t client_values_in,
                                   tosa_tensor_t client_indices,
                                   tosa_tensor_t client_input,
                                   tosa_tensor_t client_values_out,
                                   const func_ctx_t& func_ctx);

    tosa_status_t tosa_run_resize(tosa_tensor_t client_input,
                                  tosa_tensor_t client_scale,
                                  tosa_tensor_t client_offset,
                                  tosa_tensor_t client_border,
                                  const tosa_mode_t client_mode,
                                  tosa_tensor_t client_output,
                                  const func_ctx_t& func_ctx);

    tosa_status_t tosa_run_cast(tosa_tensor_t client_input, tosa_tensor_t client_output, const func_ctx_t& func_ctx);

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
                                   const func_ctx_t& func_ctx);

    tosa_status_t
        tosa_run_identity(tosa_tensor_t client_input1, tosa_tensor_t client_output, const func_ctx_t& func_ctx);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif    // OPERATORS_H_
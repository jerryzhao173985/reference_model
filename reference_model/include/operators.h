
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

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */

    // Note status needs to be aligned with graph_status
    enum tosa_status_t
    {
        tosa_status_valid         = 0,
        tosa_status_unpredictable = 1,
        tosa_status_error         = 2
    };

    enum tosa_mode_t
    {
        tosa_mode_unknown  = 0,
        tosa_mode_nearest  = 1,
        tosa_mode_bilinear = 2,
        tosa_mode_min      = 3,
        tosa_mode_max      = 4
    };

    enum tosa_datatype_t
    {
        tosa_datatype_bf16_t   = 0,
        tosa_datatype_bool_t   = 1,
        tosa_datatype_fp16_t   = 2,
        tosa_datatype_fp32_t   = 3,
        tosa_datatype_int16_t  = 4,
        tosa_datatype_int32_t  = 5,
        tosa_datatype_int48_t  = 6,
        tosa_datatype_int4_t   = 7,
        tosa_datatype_int8_t   = 8,
        tosa_datatype_uint16_t = 9,
        tosa_datatype_uint8_t  = 10,
        tosa_datatype_shape_t  = 11,
    };

    struct tosa_tensor_t
    {
        int32_t* shape;
        int32_t num_dims;
        tosa_datatype_t data_type;
        uint8_t* data;
        size_t size;
    };

    tosa_status_t tosa_run_argmax(tosa_tensor_t client_input,
                                  const int32_t client_axis,
                                  tosa_tensor_t client_output,
                                  const func_config_t& func_config = func_config_t{},
                                  const func_debug_t& func_debug   = func_debug_t{});

    tosa_status_t tosa_run_avg_pool2d(tosa_tensor_t client_input,
                                      const int32_t client_kernel[2],
                                      const int32_t client_stride[2],
                                      const int32_t client_pad[4],
                                      const int32_t client_input_zp,
                                      const int32_t client_output_zp,
                                      tosa_tensor_t client_output,
                                      const func_config_t& func_config = func_config_t{},
                                      const func_debug_t& func_debug   = func_debug_t{});

    tosa_status_t tosa_run_conv2d(tosa_tensor_t client_input,
                                  tosa_tensor_t client_weight,
                                  tosa_tensor_t client_bias,
                                  const int32_t client_pad[4],
                                  const int32_t client_stride[2],
                                  const int32_t client_dilation[2],
                                  const int32_t client_input_zp,
                                  const int32_t client_weight_zp,
                                  tosa_tensor_t client_output,
                                  const func_config_t& func_config = func_config_t{},
                                  const func_debug_t& func_debug   = func_debug_t{});

    tosa_status_t tosa_run_conv3d(tosa_tensor_t client_input,
                                  tosa_tensor_t client_weight,
                                  tosa_tensor_t client_bias,
                                  const int32_t client_pad[6],
                                  const int32_t client_stride[3],
                                  const int32_t client_dilation[3],
                                  const int32_t client_input_zp,
                                  const int32_t client_weight_zp,
                                  tosa_tensor_t client_output,
                                  const func_config_t& func_config = func_config_t{},
                                  const func_debug_t& func_debug   = func_debug_t{});

    tosa_status_t tosa_run_depthwise_conv2d(tosa_tensor_t client_input,
                                            tosa_tensor_t client_weight,
                                            tosa_tensor_t client_bias,
                                            const int32_t client_pad[4],
                                            const int32_t client_stride[2],
                                            const int32_t client_dilation[2],
                                            const int32_t client_input_zp,
                                            const int32_t client_weight_zp,
                                            tosa_tensor_t client_output,
                                            const func_config_t& func_config = func_config_t{},
                                            const func_debug_t& func_debug   = func_debug_t{});

    tosa_status_t tosa_run_fully_connected(tosa_tensor_t client_input,
                                           tosa_tensor_t client_weight,
                                           tosa_tensor_t client_bias,
                                           const int32_t client_input_zp,
                                           const int32_t client_weight_zp,
                                           tosa_tensor_t client_output,
                                           const func_config_t& func_config = func_config_t{},
                                           const func_debug_t& func_debug   = func_debug_t{});

    tosa_status_t tosa_run_matmul(tosa_tensor_t client_a,
                                  tosa_tensor_t client_b,
                                  const int32_t client_a_zp,
                                  const int32_t client_b_zp,
                                  tosa_tensor_t client_output,
                                  const func_config_t& func_config = func_config_t{},
                                  const func_debug_t& func_debug   = func_debug_t{});

    tosa_status_t tosa_run_max_pool2d(tosa_tensor_t client_input,
                                      const int32_t client_kernel[2],
                                      const int32_t client_stride[2],
                                      const int32_t client_pad[4],
                                      const int32_t client_input_zp,
                                      const int32_t client_output_zp,
                                      tosa_tensor_t client_output,
                                      const func_config_t& func_config = func_config_t{},
                                      const func_debug_t& func_debug   = func_debug_t{});

    tosa_status_t tosa_run_transpose_conv2d(tosa_tensor_t client_input,
                                            tosa_tensor_t client_weight,
                                            tosa_tensor_t client_bias,
                                            const int32_t client_stride[2],
                                            const int32_t client_input_zp,
                                            const int32_t client_weight_zp,
                                            const int32_t client_pad_len,
                                            const int32_t client_pad[],
                                            const int32_t client_dilation_len,
                                            const int32_t client_dilation[],
                                            tosa_tensor_t client_output,
                                            const func_config_t& func_config = func_config_t{},
                                            const func_debug_t& func_debug   = func_debug_t{});

    tosa_status_t tosa_run_clamp(tosa_tensor_t client_input,
                                 const int32_t client_min_int,
                                 const int32_t client_max_int,
                                 const float client_min_fp,
                                 const float client_max_fp,
                                 tosa_tensor_t client_output,
                                 const func_config_t& func_config = func_config_t{},
                                 const func_debug_t& func_debug   = func_debug_t{});

    tosa_status_t tosa_run_erf(tosa_tensor_t client_input,
                               tosa_tensor_t client_output,
                               const func_config_t& func_config = func_config_t{},
                               const func_debug_t& func_debug   = func_debug_t{});

    tosa_status_t tosa_run_sigmoid(tosa_tensor_t client_input,
                                   tosa_tensor_t client_output,
                                   const func_config_t& func_config = func_config_t{},
                                   const func_debug_t& func_debug   = func_debug_t{});

    tosa_status_t tosa_run_tanh(tosa_tensor_t client_input,
                                tosa_tensor_t client_output,
                                const func_config_t& func_config = func_config_t{},
                                const func_debug_t& func_debug   = func_debug_t{});

    tosa_status_t tosa_run_add(tosa_tensor_t client_input1,
                               tosa_tensor_t client_input2,
                               tosa_tensor_t client_output,
                               const func_config_t& func_config = func_config_t{},
                               const func_debug_t& func_debug   = func_debug_t{});

    tosa_status_t tosa_run_arithmetic_right_shift(tosa_tensor_t client_input1,
                                                  tosa_tensor_t client_input2,
                                                  const bool client_round,
                                                  tosa_tensor_t client_output,
                                                  const func_config_t& func_config = func_config_t{},
                                                  const func_debug_t& func_debug   = func_debug_t{});

    tosa_status_t tosa_run_bitwise_and(tosa_tensor_t client_input1,
                                       tosa_tensor_t client_input2,
                                       tosa_tensor_t client_output,
                                       const func_config_t& func_config = func_config_t{},
                                       const func_debug_t& func_debug   = func_debug_t{});

    tosa_status_t tosa_run_bitwise_or(tosa_tensor_t client_input1,
                                      tosa_tensor_t client_input2,
                                      tosa_tensor_t client_output,
                                      const func_config_t& func_config = func_config_t{},
                                      const func_debug_t& func_debug   = func_debug_t{});

    tosa_status_t tosa_run_bitwise_xor(tosa_tensor_t client_input1,
                                       tosa_tensor_t client_input2,
                                       tosa_tensor_t client_output,
                                       const func_config_t& func_config = func_config_t{},
                                       const func_debug_t& func_debug   = func_debug_t{});

    tosa_status_t tosa_run_intdiv(tosa_tensor_t client_input1,
                                  tosa_tensor_t client_input2,
                                  tosa_tensor_t client_output,
                                  const func_config_t& func_config = func_config_t{},
                                  const func_debug_t& func_debug   = func_debug_t{});

    tosa_status_t tosa_run_logical_and(tosa_tensor_t client_input1,
                                       tosa_tensor_t client_input2,
                                       tosa_tensor_t client_output,
                                       const func_config_t& func_config = func_config_t{},
                                       const func_debug_t& func_debug   = func_debug_t{});

    tosa_status_t tosa_run_logical_left_shift(tosa_tensor_t client_input1,
                                              tosa_tensor_t client_input2,
                                              tosa_tensor_t client_output,
                                              const func_config_t& func_config = func_config_t{},
                                              const func_debug_t& func_debug   = func_debug_t{});

    tosa_status_t tosa_run_logical_right_shift(tosa_tensor_t client_input1,
                                               tosa_tensor_t client_input2,
                                               tosa_tensor_t client_output,
                                               const func_config_t& func_config = func_config_t{},
                                               const func_debug_t& func_debug   = func_debug_t{});

    tosa_status_t tosa_run_logical_or(tosa_tensor_t client_input1,
                                      tosa_tensor_t client_input2,
                                      tosa_tensor_t client_output,
                                      const func_config_t& func_config = func_config_t{},
                                      const func_debug_t& func_debug   = func_debug_t{});

    tosa_status_t tosa_run_logical_xor(tosa_tensor_t client_input1,
                                       tosa_tensor_t client_input2,
                                       tosa_tensor_t client_output,
                                       const func_config_t& func_config = func_config_t{},
                                       const func_debug_t& func_debug   = func_debug_t{});

    tosa_status_t tosa_run_maximum(tosa_tensor_t client_input1,
                                   tosa_tensor_t client_input2,
                                   tosa_tensor_t client_output,
                                   const func_config_t& func_config = func_config_t{},
                                   const func_debug_t& func_debug   = func_debug_t{});

    tosa_status_t tosa_run_minimum(tosa_tensor_t client_input1,
                                   tosa_tensor_t client_input2,
                                   tosa_tensor_t client_output,
                                   const func_config_t& func_config = func_config_t{},
                                   const func_debug_t& func_debug   = func_debug_t{});

    tosa_status_t tosa_run_mul(tosa_tensor_t client_input1,
                               tosa_tensor_t client_input2,
                               const int32_t client_shift,
                               tosa_tensor_t client_output,
                               const func_config_t& func_config = func_config_t{},
                               const func_debug_t& func_debug   = func_debug_t{});

    tosa_status_t tosa_run_pow(tosa_tensor_t client_input1,
                               tosa_tensor_t client_input2,
                               tosa_tensor_t client_output,
                               const func_config_t& func_config = func_config_t{},
                               const func_debug_t& func_debug   = func_debug_t{});

    tosa_status_t tosa_run_sub(tosa_tensor_t client_input1,
                               tosa_tensor_t client_input2,
                               tosa_tensor_t client_output,
                               const func_config_t& func_config = func_config_t{},
                               const func_debug_t& func_debug   = func_debug_t{});

    tosa_status_t tosa_run_table(tosa_tensor_t client_input,
                                 const int32_t client_table_len,
                                 const int16_t client_table[],
                                 tosa_tensor_t client_output,
                                 const func_config_t& func_config = func_config_t{},
                                 const func_debug_t& func_debug   = func_debug_t{});

    tosa_status_t tosa_run_abs(tosa_tensor_t client_input1,
                               tosa_tensor_t client_output,
                               const func_config_t& func_config = func_config_t{},
                               const func_debug_t& func_debug   = func_debug_t{});

    tosa_status_t tosa_run_bitwise_not(tosa_tensor_t client_input1,
                                       tosa_tensor_t client_output,
                                       const func_config_t& func_config = func_config_t{},
                                       const func_debug_t& func_debug   = func_debug_t{});

    tosa_status_t tosa_run_ceil(tosa_tensor_t client_input1,
                                tosa_tensor_t client_output,
                                const func_config_t& func_config = func_config_t{},
                                const func_debug_t& func_debug   = func_debug_t{});

    tosa_status_t tosa_run_clz(tosa_tensor_t client_input1,
                               tosa_tensor_t client_output,
                               const func_config_t& func_config = func_config_t{},
                               const func_debug_t& func_debug   = func_debug_t{});

    tosa_status_t tosa_run_exp(tosa_tensor_t client_input1,
                               tosa_tensor_t client_output,
                               const func_config_t& func_config = func_config_t{},
                               const func_debug_t& func_debug   = func_debug_t{});

    tosa_status_t tosa_run_floor(tosa_tensor_t client_input1,
                                 tosa_tensor_t client_output,
                                 const func_config_t& func_config = func_config_t{},
                                 const func_debug_t& func_debug   = func_debug_t{});

    tosa_status_t tosa_run_log(tosa_tensor_t client_input1,
                               tosa_tensor_t client_output,
                               const func_config_t& func_config = func_config_t{},
                               const func_debug_t& func_debug   = func_debug_t{});

    tosa_status_t tosa_run_logical_not(tosa_tensor_t client_input1,
                                       tosa_tensor_t client_output,
                                       const func_config_t& func_config = func_config_t{},
                                       const func_debug_t& func_debug   = func_debug_t{});

    tosa_status_t tosa_run_negate(tosa_tensor_t client_input1,
                                  const int32_t client_input1_zp,
                                  const int32_t client_output_zp,
                                  tosa_tensor_t client_output,
                                  const func_config_t& func_config = func_config_t{},
                                  const func_debug_t& func_debug   = func_debug_t{});

    tosa_status_t tosa_run_reciprocal(tosa_tensor_t client_input1,
                                      tosa_tensor_t client_output,
                                      const func_config_t& func_config = func_config_t{},
                                      const func_debug_t& func_debug   = func_debug_t{});

    tosa_status_t tosa_run_rsqrt(tosa_tensor_t client_input1,
                                 tosa_tensor_t client_output,
                                 const func_config_t& func_config = func_config_t{},
                                 const func_debug_t& func_debug   = func_debug_t{});

    tosa_status_t tosa_run_select(tosa_tensor_t client_input1,
                                  tosa_tensor_t client_input2,
                                  tosa_tensor_t client_input3,
                                  tosa_tensor_t client_output,
                                  const func_config_t& func_config = func_config_t{},
                                  const func_debug_t& func_debug   = func_debug_t{});

    tosa_status_t tosa_run_equal(tosa_tensor_t client_input1,
                                 tosa_tensor_t client_input2,
                                 tosa_tensor_t client_output,
                                 const func_config_t& func_config = func_config_t{},
                                 const func_debug_t& func_debug   = func_debug_t{});

    tosa_status_t tosa_run_greater(tosa_tensor_t client_input1,
                                   tosa_tensor_t client_input2,
                                   tosa_tensor_t client_output,
                                   const func_config_t& func_config = func_config_t{},
                                   const func_debug_t& func_debug   = func_debug_t{});

    tosa_status_t tosa_run_greater_equal(tosa_tensor_t client_input1,
                                         tosa_tensor_t client_input2,
                                         tosa_tensor_t client_output,
                                         const func_config_t& func_config = func_config_t{},
                                         const func_debug_t& func_debug   = func_debug_t{});

    tosa_status_t tosa_run_reduce_all(tosa_tensor_t client_input,
                                      const int32_t client_axis,
                                      tosa_tensor_t client_output,
                                      const func_config_t& func_config = func_config_t{},
                                      const func_debug_t& func_debug   = func_debug_t{});

    tosa_status_t tosa_run_reduce_any(tosa_tensor_t client_input,
                                      const int32_t client_axis,
                                      tosa_tensor_t client_output,
                                      const func_config_t& func_config = func_config_t{},
                                      const func_debug_t& func_debug   = func_debug_t{});

    tosa_status_t tosa_run_reduce_max(tosa_tensor_t client_input,
                                      const int32_t client_axis,
                                      tosa_tensor_t client_output,
                                      const func_config_t& func_config = func_config_t{},
                                      const func_debug_t& func_debug   = func_debug_t{});

    tosa_status_t tosa_run_reduce_min(tosa_tensor_t client_input,
                                      const int32_t client_axis,
                                      tosa_tensor_t client_output,
                                      const func_config_t& func_config = func_config_t{},
                                      const func_debug_t& func_debug   = func_debug_t{});

    tosa_status_t tosa_run_reduce_product(tosa_tensor_t client_input,
                                          const int32_t client_axis,
                                          tosa_tensor_t client_output,
                                          const func_config_t& func_config = func_config_t{},
                                          const func_debug_t& func_debug   = func_debug_t{});

    tosa_status_t tosa_run_reduce_sum(tosa_tensor_t client_input,
                                      const int32_t client_axis,
                                      tosa_tensor_t client_output,
                                      const func_config_t& func_config = func_config_t{},
                                      const func_debug_t& func_debug   = func_debug_t{});

    tosa_status_t tosa_run_concat(tosa_tensor_t client_input1,
                                  const int32_t client_axis,
                                  tosa_tensor_t client_output,
                                  const func_config_t& func_config = func_config_t{},
                                  const func_debug_t& func_debug   = func_debug_t{});

    tosa_status_t tosa_run_pad(tosa_tensor_t client_input1,
                               const int32_t client_padding_len,
                               const int32_t client_padding[],
                               const int32_t client_pad_const_int,
                               const float client_pad_const_fp,
                               tosa_tensor_t client_output,
                               const func_config_t& func_config = func_config_t{},
                               const func_debug_t& func_debug   = func_debug_t{});

    tosa_status_t tosa_run_dim(tosa_tensor_t client_input1,
                               const int32_t client_axis,
                               tosa_tensor_t client_output,
                               const func_config_t& func_config = func_config_t{},
                               const func_debug_t& func_debug   = func_debug_t{});

    tosa_status_t tosa_run_reshape(tosa_tensor_t client_input1,
                                   const int32_t client_new_shape_len,
                                   const int32_t client_new_shape[],
                                   tosa_tensor_t client_output,
                                   const func_config_t& func_config = func_config_t{},
                                   const func_debug_t& func_debug   = func_debug_t{});

    tosa_status_t tosa_run_reverse(tosa_tensor_t client_input,
                                   const int32_t client_axis,
                                   tosa_tensor_t client_output,
                                   const func_config_t& func_config = func_config_t{},
                                   const func_debug_t& func_debug   = func_debug_t{});

    tosa_status_t tosa_run_slice(tosa_tensor_t client_input1,
                                 const int32_t client_start_len,
                                 const int32_t client_start[],
                                 const int32_t client_size_len,
                                 const int32_t client_size[],
                                 tosa_tensor_t client_output,
                                 const func_config_t& func_config = func_config_t{},
                                 const func_debug_t& func_debug   = func_debug_t{});

    tosa_status_t tosa_run_tile(tosa_tensor_t client_input1,
                                const int32_t client_multiples_len,
                                const int32_t client_multiples[],
                                tosa_tensor_t client_output,
                                const func_config_t& func_config = func_config_t{},
                                const func_debug_t& func_debug   = func_debug_t{});

    tosa_status_t tosa_run_transpose(tosa_tensor_t client_input1,
                                     const int32_t client_perms_len,
                                     const int32_t client_perms[],
                                     tosa_tensor_t client_output,
                                     const func_config_t& func_config = func_config_t{},
                                     const func_debug_t& func_debug   = func_debug_t{});

    tosa_status_t tosa_run_gather(tosa_tensor_t client_values,
                                  tosa_tensor_t client_indices,
                                  tosa_tensor_t client_output,
                                  const func_config_t& func_config = func_config_t{},
                                  const func_debug_t& func_debug   = func_debug_t{});

    tosa_status_t tosa_run_scatter(tosa_tensor_t client_values_in,
                                   tosa_tensor_t client_indices,
                                   tosa_tensor_t client_input,
                                   tosa_tensor_t client_values_out,
                                   const func_config_t& func_config = func_config_t{},
                                   const func_debug_t& func_debug   = func_debug_t{});

    tosa_status_t tosa_run_resize(tosa_tensor_t client_input,
                                  const int16_t client_scale[4],
                                  const int16_t client_offset[2],
                                  const int16_t client_border[2],
                                  const tosa_mode_t client_mode,
                                  tosa_tensor_t client_output,
                                  const func_config_t& func_config = func_config_t{},
                                  const func_debug_t& func_debug   = func_debug_t{});

    tosa_status_t tosa_run_cast(tosa_tensor_t client_input,
                                tosa_tensor_t client_output,
                                const func_config_t& func_config = func_config_t{},
                                const func_debug_t& func_debug   = func_debug_t{});

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
                                   const func_config_t& func_config = func_config_t{},
                                   const func_debug_t& func_debug   = func_debug_t{});

    tosa_status_t tosa_run_identity(tosa_tensor_t client_input1,
                                    tosa_tensor_t client_output,
                                    const func_config_t& func_config = func_config_t{},
                                    const func_debug_t& func_debug   = func_debug_t{});

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif    // OPERATORS_H_

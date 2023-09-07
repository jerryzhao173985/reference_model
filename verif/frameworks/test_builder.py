# Copyright (c) 2020-2023, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import tensorflow as tf
from frameworks.tensor_gen import TGen


class TBuilder:
    """The member functions build the tensorflow operators into small networks
    for our tests"""

    def __init__(self):
        pass

    def fake_quant(tensor, tensor_scale, name):
        """Helper function for quantizing with a scaling parameters structure."""
        return tf.quantization.fake_quant_with_min_max_args(
            tensor,
            min=tensor_scale.min,
            max=tensor_scale.max,
            num_bits=tensor_scale.num_bits,
            narrow_range=tensor_scale.narrow_range,
            name=name,
        )

    def fake_quant_params(tensor, min, max, scaling, name):
        """Helper function for quantizing with individual scaling parameters."""
        return tf.quantization.fake_quant_with_min_max_args(
            tensor,
            min=min,
            max=max,
            num_bits=scaling.num_bits,
            narrow_range=scaling.narrow_range,
            name=name,
        )

    class Add:
        def __init__(self, name):
            self.result_name = name

        def eval(self, a, b):
            return tf.add(a, b, name=self.result_name)

    class Sub:
        def __init__(self, name):
            self.result_name = name

        def eval(self, a, b):
            return tf.subtract(a, b, name=self.result_name)

    class Mul:
        def __init__(self, name):
            self.result_name = name

        def eval(self, a, b):
            return tf.multiply(a, b, name=self.result_name)

    class Exp:
        def __init__(self, name):
            self.result_name = name

        def eval(self, a):
            return tf.exp(a, name=self.result_name)

    class Rcp:
        def __init__(self, name):
            self.result_name = name

        def eval(self, a):
            return tf.math.reciprocal(a, name=self.result_name)

    class Relu:
        def __init__(self, name):
            self.result_name = name

        def eval(self, a):
            return tf.nn.relu(a, name=self.result_name)

    class Relu1:
        def __init__(self, name):
            self.result_name = name

        def eval(self, a):
            # TF doesn't have relu_n1_to_1 operator,
            # use min and max as a workaround
            # alternatively, we can use clip_by_value
            return tf.math.minimum(1.0, tf.math.maximum(-1.0, a))

    class Relu0To1:
        def __init__(self, name):
            self.result_name = name

        def eval(self, a):
            # TF doesn't have relu_0_to_1 operator,
            # use min and max as a workaround
            # alternatively, we can use clip_by_value
            return tf.math.minimum(1.0, tf.math.maximum(0.0, a))

    class Relu6:
        def __init__(self, name):
            self.result_name = name

        def eval(self, a):
            return tf.nn.relu6(a, name=self.result_name)

    class LeakyRelu:
        def __init__(self, alpha, name):
            self.alpha = alpha
            self.result_name = name

        def eval(self, a):
            return tf.nn.leaky_relu(a, alpha=self.alpha, name=self.result_name)

    class Prelu:
        def __init__(self, name):
            self.result_name = name
            self.prelu = tf.keras.layers.PReLU(
                alpha_initializer=tf.keras.initializers.RandomNormal(
                    mean=0.0, stddev=1.0
                )
            )

        def eval(self, a):
            return self.prelu(a)

    class Gelu:
        def __init__(self, name):
            self.result_name = name

        def eval(self, a):
            return tf.nn.gelu(a, name=self.result_name)

    class Concat:
        def __init__(self, axis, name):
            self.axis = axis
            self.result_name = name

        def eval(self, a, b):
            return tf.concat([a, b], self.axis, name=self.result_name)

    class BitwiseAnd:
        def __init__(self, name):
            self.result_name = name

        def eval(self, a, b):
            return tf.bitwise.bitwise_and(a, b, name=self.result_name)

    class BitwiseOr:
        def __init__(self, name):
            self.result_name = name

        def eval(self, a, b):
            return tf.bitwise.bitwise_or(a, b, name=self.result_name)

    class BitwiseNot:
        def __init__(self, name):
            self.result_name = name

        def eval(self, a):
            return tf.bitwise.invert(a, name=self.result_name)

    class BitwiseXor:
        def __init__(self, name):
            self.result_name = name

        def eval(self, a, b):
            return tf.bitwise.bitwise_xor(a, b, name=self.result_name)

    class LogicalAnd:
        def __init__(self, name):
            self.result_name = name

        def eval(self, a, b):
            return tf.math.logical_and(a, b, name=self.result_name)

    class LogicalOr:
        def __init__(self, name):
            self.result_name = name

        def eval(self, a, b):
            return tf.math.logical_or(a, b, name=self.result_name)

    class LogicalNot:
        def __init__(self, name):
            self.result_name = name

        def eval(self, a):
            return tf.math.logical_not(a, name=self.result_name)

    class ReduceAny:
        def __init__(self, axis_list, keepdims, name):
            self.axis_list = axis_list
            self.keepdims = keepdims
            self.result_name = name

        def eval(self, a):
            return tf.math.reduce_any(
                a, self.axis_list, keepdims=self.keepdims, name=self.result_name
            )

    class ReduceAll:
        def __init__(self, axis_list, keepdims, name):
            self.axis_list = axis_list
            self.keepdims = keepdims
            self.result_name = name

        def eval(self, a):
            return tf.math.reduce_all(
                a, self.axis_list, keepdims=self.keepdims, name=self.result_name
            )

    class ReduceMin:
        def __init__(self, axis_list, keepdims, name):
            self.axis_list = axis_list
            self.keepdims = keepdims
            self.result_name = name

        def eval(self, a):
            return tf.math.reduce_min(
                a, self.axis_list, keepdims=self.keepdims, name=self.result_name
            )

    class ReduceMax:
        def __init__(self, axis_list, keepdims, name):
            self.axis_list = axis_list
            self.keepdims = keepdims
            self.result_name = name

        def eval(self, a):
            return tf.math.reduce_max(
                a, self.axis_list, keepdims=self.keepdims, name=self.result_name
            )

    class ReduceSum:
        def __init__(self, axis_list, keepdims, name):
            self.axis_list = axis_list
            self.keepdims = keepdims
            self.result_name = name

        def eval(self, a):
            return tf.math.reduce_sum(
                a, self.axis_list, keepdims=self.keepdims, name=self.result_name
            )

    class ReduceMean:
        def __init__(self, axis_list, keepdims, name):
            self.axis_list = axis_list
            self.keepdims = keepdims
            self.result_name = name

        def eval(self, a):
            return tf.math.reduce_mean(
                a, self.axis_list, keepdims=self.keepdims, name=self.result_name
            )

    class ReduceProduct:
        def __init__(self, axis_list, keepdims, name):
            self.axis_list = axis_list
            self.keepdims = keepdims
            self.result_name = name

        def eval(self, a):
            return tf.math.reduce_prod(
                a, self.axis_list, keepdims=self.keepdims, name=self.result_name
            )

    class Min:
        def __init__(self, name):
            self.result_name = name

        def eval(self, a, b):
            return tf.math.minimum(a, b, name=self.result_name)

    class Max:
        def __init__(self, name):
            self.result_name = name

        def eval(self, a, b):
            return tf.math.maximum(a, b, name=self.result_name)

    class Pow:
        def __init__(self, name):
            self.result_name = name

        def eval(self, a, b):
            return tf.math.pow(a, b, name=self.result_name)

    class Abs:
        def __init__(self, name):
            self.result_name = name

        def eval(self, a):
            return tf.math.abs(a, name=self.result_name)

    class Ceil:
        def __init__(self, name):
            self.result_name = name

        def eval(self, a):
            return tf.math.ceil(a, name=self.result_name)

    class Floor:
        def __init__(self, name):
            self.result_name = name

        def eval(self, a):
            return tf.math.floor(a, name=self.result_name)

    class Log:
        def __init__(self, name):
            self.result_name = name

        def eval(self, a):
            return tf.math.log(a, name=self.result_name)

    class Negate:
        def __init__(self, name):
            self.result_name = name

        def eval(self, a):
            return tf.math.negative(a, name=self.result_name)

    class Rsqrt:
        def __init__(self, name):
            self.result_name = name

        def eval(self, a):
            return tf.math.rsqrt(a, name=self.result_name)

    class Sign:
        def __init__(self, name):
            self.result_name = name

        def eval(self, a):
            return tf.math.sign(a, name=self.result_name)

    class Sigmoid:
        def __init__(self, name):
            self.result_name = name

        def eval(self, a):
            return tf.math.sigmoid(a, name=self.result_name)

    class Tanh:
        def __init__(self, name):
            self.result_name = name

        def eval(self, a):
            return tf.math.tanh(a, name=self.result_name)

    class Erf:
        # tfl.ops cannot be generated right now.
        # https://github.com/tensorflow/tensorflow/issues/60809
        def __init__(self, name):
            self.result_name = name

        def eval(self, a):
            return tf.math.erf(a, name=self.result_name)

    class Sin:
        def __init__(self, name):
            self.result_name = name

        def eval(self, a):
            return tf.math.sin(a, name=self.result_name)

    class Cos:
        def __init__(self, name):
            self.result_name = name

        def eval(self, a):
            return tf.math.cos(a, name=self.result_name)

    class Atan2:
        def __init__(self, name):
            self.result_name = name

        def eval(self, a, b):
            return tf.math.atan2(a, b, name=self.result_name)

    class Square:
        def __init__(self, name):
            self.result_name = name

        def eval(self, a):
            return tf.math.square(a, name=self.result_name)

    class SquaredDifference:
        def __init__(self, name):
            self.result_name = name

        def eval(self, a, b):
            return tf.math.squared_difference(a, b, name=self.result_name)

    class Equal:
        def __init__(self, name):
            self.result_name = name

        def eval(self, a, b):
            return tf.math.equal(a, b, name=self.result_name)

    class GreaterEqual:
        def __init__(self, name):
            self.result_name = name

        def eval(self, a, b):
            return tf.math.greater_equal(a, b, name=self.result_name)

    class Greater:
        def __init__(self, name):
            self.result_name = name

        def eval(self, a, b):
            return tf.math.greater(a, b, name=self.result_name)

    class Less:
        def __init__(self, name):
            self.result_name = name

        def eval(self, a, b):
            return tf.math.less(a, b, name=self.result_name)

    class LessEqual:
        def __init__(self, name):
            self.result_name = name

        def eval(self, a, b):
            return tf.math.less_equal(a, b, name=self.result_name)

    class Conv2d:
        def __init__(self, weight, strides, padding, dilations, name):
            self.weight = weight
            self.strides = strides
            self.padding = padding
            self.dilations = dilations
            self.result_name = name

        def eval(self, input):
            return tf.nn.conv2d(
                input,
                self.weight,
                self.strides,
                self.padding,
                data_format="NHWC",
                dilations=self.dilations,
                name=self.result_name,
            )

    class Conv2dRelu:
        def __init__(self, weight, name):
            self.weight = weight
            self.result_name = name

        def eval(self, input):
            conv2d = tf.nn.conv2d(
                input,
                self.weight,
                [1, 1, 1, 1],
                "SAME",
                data_format="NHWC",
                dilations=[1, 1, 1, 1],
                name="conv2d",
            )
            return tf.nn.relu(conv2d, name=self.result_name)

    class Conv2dRelu6:
        def __init__(self, weight, name):
            self.weight = weight
            self.result_name = name

        def eval(self, input):
            conv2d = tf.nn.conv2d(
                input,
                self.weight,
                [1, 1, 1, 1],
                "SAME",
                data_format="NHWC",
                dilations=[1, 1, 1, 1],
                name="conv2d",
            )
            return tf.nn.relu6(conv2d, name=self.result_name)

    class Conv2dReluN1To1:
        def __init__(self, weight, name):
            self.weight = weight
            self.result_name = name

        def eval(self, input):
            conv2d = tf.nn.conv2d(
                input,
                self.weight,
                [1, 1, 1, 1],
                "SAME",
                data_format="NHWC",
                dilations=[1, 1, 1, 1],
                name="conv2d",
            )
            return tf.clip_by_value(conv2d, -1.0, 1.0, name=self.result_name)

    class Conv2dTanh:
        def __init__(self, weight, name):
            self.weight = weight
            self.result_name = name

        def eval(self, input):
            conv2d = tf.nn.conv2d(
                input,
                self.weight,
                [1, 1, 1, 1],
                "SAME",
                data_format="NHWC",
                dilations=[1, 1, 1, 1],
                name="conv2d",
            )
            return tf.math.tanh(conv2d, name=self.result_name)

    class Conv2dWithBias:
        def __init__(self, weight, bias, strides, padding, dilations, name):
            self.weight = weight
            self.bias = bias
            self.strides = strides
            self.padding = padding
            self.dilations = dilations
            self.result_name = name

        def eval(self, input):
            conv2d_op = tf.nn.conv2d(
                input,
                self.weight,
                self.strides,
                self.padding,
                data_format="NHWC",
                dilations=self.dilations,
                name="conv2d",
            )
            bias_add_op = tf.nn.bias_add(
                conv2d_op, self.bias, data_format="NHWC", name=self.result_name
            )
            return bias_add_op

    class Conv3d:
        def __init__(self, weight, strides, padding, dilations, name):
            self.weight = weight
            self.strides = strides
            self.padding = padding
            self.dilations = dilations
            self.result_name = name

        def eval(self, input):
            return tf.nn.conv3d(
                input,
                self.weight,
                self.strides,
                self.padding,
                data_format="NDHWC",
                dilations=self.dilations,
                name=self.result_name,
            )

    class Conv3dWithBias:
        def __init__(self, weight, bias, strides, padding, dilations, name):
            self.weight = weight
            self.bias = bias
            self.strides = strides
            self.padding = padding
            self.dilations = dilations
            self.result_name = name

        def eval(self, input):
            conv3d_op = tf.nn.conv3d(
                input,
                self.weight,
                self.strides,
                self.padding,
                data_format="NDHWC",
                dilations=self.dilations,
                name="conv3d",
            )
            bias_add_op = tf.nn.bias_add(conv3d_op, self.bias, name=self.result_name)
            return bias_add_op

    class DepthwiseConv2d:
        def __init__(self, weight, strides, padding, dilations, name):
            self.weight = weight
            self.strides = strides
            self.padding = padding
            self.dilations = dilations
            self.result_name = name

        def eval(self, input):
            dws_conv2d = tf.nn.depthwise_conv2d(
                input,
                self.weight,
                self.strides,
                self.padding,
                data_format="NHWC",
                dilations=self.dilations,
                name="dws_conv2d",
            )
            return tf.identity(dws_conv2d, name=self.result_name)

    class DepthwiseConv2dWithBias:
        def __init__(self, weight, bias, strides, padding, dilations, name):
            self.weight = weight
            self.bias = bias
            self.strides = strides
            self.padding = padding
            self.dilations = dilations
            self.result_name = name

        def eval(self, input):
            dws_conv2d = tf.nn.depthwise_conv2d(
                input,
                self.weight,
                self.strides,
                self.padding,
                data_format="NHWC",
                dilations=self.dilations,
                name="dws_conv2d",
            )
            bias_add_op = tf.nn.bias_add(
                dws_conv2d, self.bias, data_format="NHWC", name=self.result_name
            )
            return bias_add_op

    class TransposeConv2d:
        def __init__(self, weight, output_shape, strides, padding, name):
            self.weight = weight
            self.output_shape = output_shape
            self.strides = strides
            self.padding = padding
            self.result_name = name

        def eval(self, input):
            return tf.nn.conv2d_transpose(
                input,
                self.weight,
                self.output_shape,
                self.strides,
                self.padding,
                data_format="NHWC",
                name=self.result_name,
            )

    class Argmax:
        def __init__(self, axis, name):
            self.axis = axis
            self.result_name = name

        def eval(self, a):
            return tf.argmax(a, self.axis, output_type=tf.int32, name=self.result_name)

    class AvgPool2d:
        def __init__(self, strides, kernel_size, padding, name):
            self.strides = strides
            self.kernel_size = kernel_size
            self.padding = padding
            self.result_name = name

        def eval(self, input):
            return tf.nn.avg_pool2d(
                input,
                strides=self.strides,
                ksize=self.kernel_size,
                padding=self.padding,
                data_format="NHWC",
                name=self.result_name,
            )

    class MaxPool2d:
        def __init__(self, strides, kernel_size, padding, name):
            self.strides = strides
            self.kernel_size = kernel_size
            self.padding = padding
            self.result_name = name

        def eval(self, input):
            return tf.nn.max_pool2d(
                input,
                strides=self.strides,
                ksize=self.kernel_size,
                padding=self.padding,
                data_format="NHWC",
                name=self.result_name,
            )

    class Reshape:
        def __init__(self, shape, name):
            self.shape = shape
            self.result_name = name

        def eval(self, a):
            reshape_op = tf.reshape(a, self.shape)
            return tf.identity(reshape_op, name=self.result_name)

    class Transpose:
        def __init__(self, perm, name):
            self.perm = perm
            self.result_name = name

        def eval(self, a):
            return tf.transpose(a, self.perm, name=self.result_name)

    class Slice:
        def __init__(self, begin, size, name):
            self.begin = begin
            self.size = size
            self.result_name = name

        def eval(self, a):
            return tf.slice(a, begin=self.begin, size=self.size, name=self.result_name)

    class StridedSlice:
        def __init__(
            self,
            begin,
            end,
            strides,
            begin_mask,
            end_mask,
            ellipsis_mask,
            new_axis_mask,
            shrink_axis_mask,
            name,
        ):
            self.begin = begin
            self.end = end
            self.strides = strides
            self.begin_mask = begin_mask
            self.end_mask = end_mask
            self.ellipsis_mask = ellipsis_mask
            self.new_axis_mask = new_axis_mask
            self.shrink_axis_mask = shrink_axis_mask
            self.result_name = name

        def eval(self, a):
            return tf.strided_slice(
                a,
                begin=self.begin,
                end=self.end,
                strides=self.strides,
                begin_mask=self.begin_mask,
                end_mask=self.end_mask,
                ellipsis_mask=self.ellipsis_mask,
                new_axis_mask=self.new_axis_mask,
                shrink_axis_mask=self.shrink_axis_mask,
                name=self.result_name,
            )

    class Select:
        def __init__(self, name):
            self.result_name = name

        def eval(self, selector, a, b):
            return tf.where(condition=selector, x=a, y=b, name=self.result_name)

    class Addn:
        def __init__(self, name):
            self.result_name = name

        def eval(self, a, b, c, d):
            return tf.add_n([a, b, c, d], name=self.result_name)

    class Concatv2:
        def __init__(self, axis, name):
            self.axis = axis
            self.result_name = name

        def eval(self, a, b, c, d):
            return tf.concat([a, b, c, d], axis=self.axis, name=self.result_name)

    class Stack:
        def __init__(self, axis, name):
            self.axis = axis
            self.result_name = name

        def eval(self, a, b, c, d):
            return tf.stack([a, b, c, d], axis=self.axis, name=self.result_name)

    class Unstack:
        def __init__(self, axis, name):
            self.axis = axis
            self.result_name = name

        def eval(self, a):
            unstack_op = tf.unstack(a, axis=self.axis, name="unstack_op")
            result_count = a.shape[self.axis]

            if result_count == 1:
                return tf.identity(unstack_op[0], name=self.result_name)

            sums = []
            for i in range(result_count):
                sums.append(
                    tf.math.reduce_sum(unstack_op[i], name="reduce_{}".format(i))
                )
            return tf.stack(sums, 0, name=self.result_name)

    class MirrorPad:
        def __init__(self, padding, mode, name):
            self.padding = padding
            self.mode = mode
            self.result_name = name

        def eval(self, a):
            return tf.pad(
                a,
                self.padding,
                mode=self.mode,
                constant_values=0,
                name=self.result_name,
            )

    class Pad:
        def __init__(self, padding, pad_const, name):
            self.padding = padding
            self.pad_const = pad_const
            self.result_name = name

        def eval(self, a):
            return tf.pad(
                a,
                self.padding,
                mode="CONSTANT",
                constant_values=self.pad_const,
                name=self.result_name,
            )

    class ExpandDims:
        def __init__(self, axis, name):
            self.axis = axis
            self.result_name = name

        def eval(self, a):
            return tf.expand_dims(a, self.axis, name=self.result_name)

    class Shape:
        def __init__(self, name):
            self.result_name = name

        def eval(self, a):
            return tf.shape(a, name=self.result_name)

    class Rank:
        def __init__(self, name):
            self.result_name = name

        def eval(self, a):
            return tf.rank(a, name=self.result_name)

    class Fill:
        def __init__(self, shape, value, name):
            self.shape = shape
            self.value = value
            self.result_name = name

        def eval(self, a):
            return tf.fill(self.shape, self.value, name=self.result_name)

    class Elu:
        def __init__(self, name):
            self.result_name = name

        def eval(self, a):
            return tf.nn.elu(a, name=self.result_name)

    class Softmax:
        def __init__(self, name):
            self.result_name = name

        def eval(self, a):
            return tf.nn.softmax(a, name=self.result_name)

    class LogSoftmax:
        def __init__(self, name):
            self.result_name = name

        def eval(self, a):
            return tf.nn.log_softmax(a, name=self.result_name)

    class MatMul:
        def __init__(self, name):
            self.result_name = name

        def eval(self, a, b):
            return tf.linalg.matmul(a, b, name=self.result_name)

    class AddScalar:
        def __init__(self, name):
            self.result_name = name

        def eval(self, a):
            return tf.add(a, 1, name=self.result_name)

    class Add1d:
        def __init__(self, name):
            self.result_name = name

        def eval(self, a, b):
            if len(b.shape) > 1:
                b_1d = tf.reduce_sum(b, axis=list(range(0, len(b.shape) - 1, 1)))
            else:
                b_1d = b
            return tf.add(a, b_1d, name=self.result_name)

    class Split:
        def __init__(self, num_splits, axis, name):
            self.num_splits = num_splits
            self.axis = axis
            self.result_name = name

        def eval(self, a):
            # The split op generates a list of outputs.  Since we have difficulty
            # serializing a list or array of Numpy arrays, we will reduce each of
            # the results

            if not isinstance(self.num_splits, list):
                split_op = tf.split(
                    a, num_or_size_splits=self.num_splits, axis=self.axis, name="split"
                )
                result_count = self.num_splits
            else:
                num_split = np.asarray(self.num_splits, dtype=np.int32)
                split_vec_op = tf.compat.v1.constant(
                    num_split,
                    shape=num_split.shape,
                    dtype=tf.int32,
                    name="const_split_vec",
                )
                split_op = tf.split(
                    a, num_or_size_splits=split_vec_op, axis=self.axis, name="split"
                )
                result_count = num_split.shape[0]

            sums = []
            for i in range(result_count):
                sums.append(tf.math.reduce_sum(split_op[i], name="reduce_{}".format(i)))
            return tf.stack(sums, 0, name=self.result_name)

    class Tile:
        def __init__(self, multiples, name):
            self.multiples = multiples
            self.result_name = name

        def eval(self, a):
            t = tf.tile(a, self.multiples, name="tile")
            return tf.identity(t, name=self.result_name)

    class Reverse:
        def __init__(self, axis, name):
            self.axis = axis
            self.result_name = name

        def eval(self, a):
            return tf.reverse(a, [self.axis], name=self.result_name)

    class Gather:
        def __init__(self, indices, batch_dims, axis, name):
            self.indices = indices
            self.batch_dims = batch_dims
            self.axis = axis
            self.result_name = name

        def eval(self, a):
            return tf.gather(
                a,
                self.indices,
                batch_dims=self.batch_dims,
                axis=self.axis,
                name=self.result_name,
            )

    class GatherNd:
        def __init__(self, indices, name):
            self.indices = indices
            self.result_name = name

        def eval(self, a):
            return tf.gather_nd(a, self.indices, name=self.result_name)

    class ScatterNd:
        def __init__(self, shape, indices_shape, N, rng, name):
            self.shape = shape
            self.indices_shape = indices_shape
            self.N = N
            self.rng = rng
            self.result_name = name

        def eval(self, a):

            # This operator is special.  The indices and updates tensors really need
            # to be created together, but in the current structure of this tool there
            # is no way to do that before now.  The number of updates is determined by
            # the indices, so we can really only create that after indices; but we
            # don't know the type at that time.
            #
            # Shapes are guaranteed deterministic, but we'll use our rng
            # copied from the arggen stage.  It's possible that index and
            # update *values* will be non-deterministic.
            #
            # We take the tensor_tensor simply to get the dtype.

            shape_const = tf.constant(self.shape, tf.int32)

            updates_shape = list(self.indices_shape[:-1])
            updates_shape.extend(self.shape[self.indices_shape[-1] :])

            updates_const = tf.constant(TGen.getRand(updates_shape, a.dtype, self.rng))

            indices = np.zeros(self.indices_shape, dtype=np.int32)

            # We need to generate the random indices tensor based on the
            # limits of 'shape' for each dimension.  Surely, there is a faster
            # vectorized way to do this, but the tensors are fairly small so we
            # will do this one element at a time.  Each element needs to be sized based
            # on the size of the last dimension.
            for idx in np.ndindex(indices.shape):
                indices[idx] = self.rng.integers(0, self.shape[idx[-1]], size=1)[0]
                # print('{} {}'.format(idx, indices[idx]))

            indices_const = tf.constant(indices, dtype=tf.int32)

            return tf.scatter_nd(
                indices=indices_const,
                updates=updates_const,
                shape=shape_const,
                name=self.result_name,
            )

    class SpaceToBatch:
        def __init__(self, block_shape, padding, name):
            self.block_shape = block_shape
            self.padding = padding
            self.result_name = name

        def eval(self, a):
            return tf.space_to_batch(
                a, self.block_shape, self.padding, name=self.result_name
            )

    class BatchToSpace:
        def __init__(self, block_shape, cropping, name):
            self.block_shape = block_shape
            self.cropping = cropping
            self.result_name = name

        def eval(self, a):
            # transpose to swap depth and batch first. this could avoid adding new shape
            block_rank = len(self.block_shape)
            perm = [len(a.shape) - 1]
            for i in range(block_rank):
                perm.append(i + 1)
            perm.append(0)
            transpose_op = tf.transpose(a, perm)
            return tf.batch_to_space(
                transpose_op, self.block_shape, self.cropping, name=self.result_name
            )

    class SpaceToDepth:
        def __init__(self, block_shape, name):
            self.block_shape = block_shape
            self.result_name = name

        def eval(self, a):
            return tf.nn.space_to_depth(a, self.block_shape, name=self.result_name)

    class DepthToSpace:
        def __init__(self, block_shape, name):
            self.block_shape = block_shape
            self.result_name = name

        def eval(self, a):
            return tf.nn.depth_to_space(a, self.block_shape, name=self.result_name)

    class OneHot:
        def __init__(self, depth, axis, name):
            self.depth = depth
            self.axis = axis
            self.result_name = name

        def eval(self, indices, on_value, off_value):
            return tf.one_hot(
                indices,
                self.depth,
                on_value,
                off_value,
                self.axis,
                on_value.dtype,
                self.result_name,
            )

    class Fakequant:
        def __init__(self, num_bits, narrow_range, name):
            self.num_bits = num_bits
            self.narrow_range = narrow_range
            self.result_name = name

        def eval(self, a):
            return tf.quantization.fake_quant_with_min_max_args(
                a,
                min=-2.0,
                max=2.0,
                num_bits=self.num_bits,
                narrow_range=self.narrow_range,
                name=self.result_name,
            )

    class Resize:
        def __init__(self, mode, align, half, scale, name):
            self.result_name = name
            self.mode = mode
            self.align = align
            self.half = half
            self.scale = scale

        def eval(self, a):
            out_shape = []
            out_shape.append(a.shape[1] * self.scale)
            out_shape.append(a.shape[2] * self.scale)

            tf_resize_dict = (
                {"tf_resize_func": tf.compat.v1.image.resize_nearest_neighbor}
                if (self.mode == "nearest")
                else {"tf_resize_func": tf.compat.v1.image.resize_bilinear}
            )
            resize = tf_resize_dict["tf_resize_func"](
                a,
                out_shape,
                align_corners=self.align,
                name="resize",
                half_pixel_centers=self.half,
            )
            return tf.identity(resize, name=self.result_name)

    class LeftShift:
        def __init__(self, shift, name):
            self.shift = shift
            self.result_name = name

        def eval(self, a):
            return tf.bitwise.left_shift(a, self.shift, name=self.result_name)

    class RightShift:
        def __init__(self, shift, name):
            self.shift = shift
            self.result_name = name

        def eval(self, a):
            return tf.bitwise.right_shift(a, self.shift, name=self.result_name)

    class While:
        def __init__(self, name):
            self.result_name = name

        def while_cond(self, x):
            return tf.reduce_sum(x) < self.cap

        def while_body(self, x):
            return tf.add(x, tf.math.sigmoid(x))

        def eval(self, a):
            self.cap = tf.cast(
                tf.constant(
                    2.0,
                    shape=[
                        1,
                    ],
                ),
                a.dtype,
            )

            result = tf.while_loop(
                self.while_cond, self.while_body, [a], name=self.result_name
            )

            return result[0]

    class LSTM(tf.Module):
        def __init__(self, name):
            self.result_name = name
            self.lstm = tf.keras.layers.LSTM(
                2,
                activation="tanh",
                unroll=False,
                recurrent_activation="sigmoid",
                use_bias=True,
                recurrent_initializer="ones",
                kernel_initializer="ones",
            )

        def eval(self, a):
            return self.lstm(a)

    class SLSTM(tf.Module):
        def __init__(self, name):
            self.result_name = name
            self.lstm = tf.keras.layers.LSTM(
                2,
                stateful=True,
                activation="tanh",
                unroll=False,
                recurrent_activation="sigmoid",
                use_bias=True,
                recurrent_initializer="ones",
                kernel_initializer="ones",
            )

        def eval(self, a):
            return self.lstm(a)

    class GRU:
        def __init__(self, name):
            self.result_name = name
            self.lstm = tf.keras.layers.GRU(
                2,
                recurrent_activation="sigmoid",
                use_bias=True,
                recurrent_initializer="ones",
                kernel_initializer="ones",
            )

        def eval(self, a):
            return self.lstm(a)

    class RNN:
        def __init__(self, name):
            self.result_name = name
            basic_cell = tf.keras.layers.SimpleRNNCell(
                units=2,
                activation="sigmoid",
                use_bias=True,
                recurrent_initializer="ones",
            )
            self.rnn = tf.keras.layers.RNN(basic_cell, unroll=False)

        def eval(self, a):
            return self.rnn(a)

    class FullyConnected:
        def __init__(self, name):
            self.result_name = name
            self.dense = tf.keras.layers.Dense(2)

        def eval(self, a):
            return self.dense(a)

    class RFFT2d:
        def __init__(self, fft_length, name):
            self.fft_length = fft_length
            self.result_name = name

        def eval(self, a):
            return tf.signal.rfft2d(a, self.fft_length, name=self.result_name)

    class Real:
        def __init__(self, name):
            self.result_name = name

        def eval(self, a):
            return tf.math.real(a, name=self.result_name)

    class Imag:
        def __init__(self, name):
            self.result_name = name

        def eval(self, a):
            return tf.math.imag(a, name=self.result_name)

    class BroadcastTo:
        def __init__(self, shape, name):
            self.shape = shape
            self.result_name = name

        def eval(self, a):
            return tf.broadcast_to(a, shape=self.shape, name=self.result_name)

    class CallOnce(tf.Module):
        def __init__(self, name):
            print(tf.__version__)
            self.result_name = name
            self.var = tf.Variable([1.0])

        @tf.function(
            input_signature=[
                tf.TensorSpec(
                    shape=[
                        1,
                    ],
                    dtype=tf.float32,
                )
            ]
        )
        def eval(self, a):
            return self.var.assign([2.0])

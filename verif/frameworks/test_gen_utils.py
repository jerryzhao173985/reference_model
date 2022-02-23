# Copyright (c) 2020-2022, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
from enum import IntEnum
from enum import unique

import tensorflow as tf


# Get a string name for a given shape
def get_shape_str(shape, dtype):
    shape_name = None
    for dim in shape:
        shape_name = (shape_name + "x" + str(dim)) if shape_name else str(dim)

    if dtype == tf.float32:
        shape_name = shape_name + "_f32"
    elif dtype == tf.float16:
        shape_name = shape_name + "_f16"
    elif dtype == tf.int32:
        shape_name = shape_name + "_i32"
    elif dtype == tf.uint32:
        shape_name = shape_name + "_u32"
    elif dtype == tf.bool:
        shape_name = shape_name + "_bool"
    elif dtype == tf.quint8:
        shape_name = shape_name + "_qu8"
    elif dtype == tf.qint8:
        shape_name = shape_name + "_qi8"
    elif dtype == tf.qint16:
        shape_name = shape_name + "_qi16"
    elif dtype == tf.quint16:
        shape_name = shape_name + "_qu16"
    else:
        raise Exception("Unsupported type: {}".format(dtype))

    return shape_name


@unique
class QuantType(IntEnum):
    UNKNOWN = 0
    ALL_I8 = 1
    ALL_U8 = 2
    ALL_I16 = 3
    # TODO: support QUINT16
    CONV_U8_U8 = 4
    CONV_I8_I8 = 5
    CONV_I8_I4 = 6
    CONV_I16_I8 = 7


def get_tf_dtype(quantized_inference_dtype):
    if quantized_inference_dtype == QuantType.ALL_I8:
        return tf.qint8
    elif quantized_inference_dtype == QuantType.ALL_U8:
        return tf.quint8
    elif quantized_inference_dtype == QuantType.ALL_I16:
        return tf.qint16
    elif quantized_inference_dtype == QuantType.CONV_U8_U8:
        return tf.quint8
    elif quantized_inference_dtype == QuantType.CONV_I8_I8:
        return tf.qint8
    elif quantized_inference_dtype == QuantType.CONV_I8_I4:
        return tf.qint8
    elif quantized_inference_dtype == QuantType.CONV_I16_I8:
        return tf.qint16
    else:
        return None


class TensorScale:
    def __init__(self, _min, _max, _num_bits, _narrow_range):
        self.min = _min
        self.max = _max
        self.num_bits = _num_bits
        self.narrow_range = _narrow_range

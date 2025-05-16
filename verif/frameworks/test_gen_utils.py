# Copyright (c) 2020-2025, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
from enum import Enum
from enum import IntEnum
from enum import unique

from tosa.DType import DType

try:
    import tensorflow as tf
except ImportError:
    print("Cannot import TensorFlow in `test_gen_utils`. Skipping TF/TFL tests")
    tf = None

try:
    import torch
except ImportError:
    print("Cannot import Torch in `test_gen_utils`. Skipping Torch tests")
    torch = None


# The scaling factor for random numbers generated in input tensors.  The
# random numbers are calculated as:
# (np.random.rand() - RAND_SHIFT_FACTOR) * RAND_SCALE_FACTOR
# FIXME: improve range here
RAND_SCALE_FACTOR = 4.0
# Amount to add to random numbers
RAND_SHIFT_FACTOR = 0.5

RAND_INT_MIN = -128
RAND_INT_MAX = 128


class ElemSignedness(Enum):
    ALL_RANGE = 1
    POSITIVE = 2
    NEGATIVE = 3


DTYPE_ATTRIBUTES = {
    DType.BOOL: {"str": "b", "width": 1},
    DType.INT4: {"str": "i4", "width": 4},
    DType.INT8: {"str": "i8", "width": 8},
    DType.INT16: {"str": "i16", "width": 16},
    DType.INT32: {"str": "i32", "width": 32},
    DType.INT48: {"str": "i48", "width": 48},
    DType.FP16: {"str": "f16", "width": 16},
    DType.BF16: {"str": "bf16", "width": 16},
    DType.FP32: {"str": "f32", "width": 32},
}


# Get a string name for a given shape
def get_shape_str(shape, dtype):
    shape_name = None
    if len(shape) == 0:
        shape_name = "0"

    for dim in shape:
        shape_name = (shape_name + "x" + str(dim)) if shape_name else str(dim)

    suffix_map = {}
    if tf is not None:
        suffix_map.update(
            {
                tf.float32: "_f32",
                tf.float16: "_f16",
                tf.int8: "_i8",
                tf.int16: "_i16",
                tf.int32: "_i32",
                tf.uint32: "_u32",
                tf.bool: "_bool",
                tf.quint8: "_qu8",
                tf.qint8: "_qi8",
                tf.qint16: "_qi16",
                tf.quint16: "_qu16",
                tf.complex64: "_c64",
            }
        )
    if torch is not None:
        suffix_map.update(
            {
                torch.float32: "_f32",
                torch.float16: "_f16",
                torch.int32: "_i32",
                torch.bool: "_bool",
            }
        )

    try:
        suffix = suffix_map[dtype]
    except KeyError:
        raise Exception("Unsupported type: {}".format(dtype))

    return shape_name + suffix


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


def get_torch_dtype(quantized_inference_dtype):
    if quantized_inference_dtype == QuantType.ALL_I8:
        return torch.qint8
    elif quantized_inference_dtype == QuantType.ALL_U8:
        return torch.quint8
    elif quantized_inference_dtype == QuantType.CONV_U8_U8:
        return torch.quint8
    elif quantized_inference_dtype == QuantType.CONV_I8_I8:
        return torch.qint8
    elif quantized_inference_dtype == QuantType.CONV_I8_I4:
        return torch.qint8
    else:
        return None


class TensorScale:
    def __init__(self, _min, _max, _num_bits, _narrow_range):
        self.min = _min
        self.max = _max
        self.num_bits = _num_bits
        self.narrow_range = _narrow_range

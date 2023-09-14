# Copyright (c) 2021-2023, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
import struct
import sys
from enum import IntEnum

import numpy as np
from tosa.DType import DType

# Maximum dimension size for output and inputs for RESIZE
MAX_RESIZE_DIMENSION = 16384

# Data type information dictionary
# - str: filename abbreviation
# - width: number of bytes needed for type
# - json: JSON type string
DTYPE_ATTRIBUTES = {
    DType.BOOL: {"str": "b", "width": 1, "json": "BOOL"},
    DType.INT4: {"str": "i4", "width": 4, "json": "INT4"},
    DType.INT8: {"str": "i8", "width": 8, "json": "INT8"},
    DType.UINT8: {"str": "u8", "width": 8, "json": "UINT8"},
    DType.INT16: {"str": "i16", "width": 16, "json": "INT16"},
    DType.UINT16: {"str": "u16", "width": 16, "json": "UINT16"},
    DType.INT32: {"str": "i32", "width": 32, "json": "INT32"},
    DType.INT48: {"str": "i48", "width": 48, "json": "INT48"},
    DType.SHAPE: {"str": "s", "width": 64, "json": "SHAPE"},
    DType.FP16: {"str": "f16", "width": 16, "json": "FP16"},
    DType.BF16: {"str": "bf16", "width": 16, "json": "BF16"},
    DType.FP32: {"str": "f32", "width": 32, "json": "FP32"},
}


class ComplianceMode(IntEnum):
    """Compliance mode types."""

    EXACT = 0
    DOT_PRODUCT = 1
    ULP = 2
    FP_SPECIAL = 3
    REDUCE_PRODUCT = 4
    ROUND = 5


class DataGenType(IntEnum):
    """Data generator types."""

    PSEUDO_RANDOM = 0
    DOT_PRODUCT = 1
    OP_BOUNDARY = 2
    OP_FULLSET = 3
    OP_SPECIAL = 4


# Additional (optional) data for dot product data generator
DG_DOT_PRODUCT_OPTIONAL_INFO = ("acc_type", "kernel", "axis")


def dtypeIsFloat(dtype):
    return dtype in (DType.FP16, DType.BF16, DType.FP32)


def valueToName(item, value):
    """Get the name of an attribute with the given value.

    This convenience function is needed to print meaningful names for
    the values of the tosa.Op.Op and tosa.DType.DType classes.
    This would not be necessary if they were subclasses of Enum, or
    IntEnum, which, sadly, they are not.

    Args:
        item: The class, or object, to find the value in
        value: The value to find

    Example, to get the name of a DType value:

        name = valueToName(DType, DType.INT8)   # returns 'INT8'
        name = valueToName(DType, 4)            # returns 'INT8'

    Returns:
        The name of the first attribute found with a matching value,

    Raises:
        ValueError if the value is not found
    """
    for attr in dir(item):
        if getattr(item, attr) == value:
            return attr
    raise ValueError(f"value ({value}) not found")


def allDTypes(*, excludes=None):
    """Get a set of all DType values, optionally excluding some values.

    This convenience function is needed to provide a sequence of DType values.
    This would be much easier if DType was a subclass of Enum, or IntEnum,
    as we could then iterate over the values directly, instead of using
    dir() to find the attributes and then check if they are what we want.

    Args:
        excludes: iterable of DTYPE values (e.g. [DType.INT8, DType.BOOL])

    Returns:
        A set of DType values
    """
    excludes = () if not excludes else excludes
    return {
        getattr(DType, t)
        for t in dir(DType)
        if not callable(getattr(DType, t))
        and not t.startswith("__")
        and getattr(DType, t) not in excludes
    }


def usableDTypes(*, excludes=None):
    """Get a set of usable DType values, optionally excluding some values.

    Excludes uncommon types (DType.UNKNOWN, DType.UINT16, DType.UINT8) in
    addition to the excludes specified by the caller, as the serializer lib
    does not support them.
    If you wish to include 'UNKNOWN', 'UINT8' or 'UINT16' use allDTypes
    instead.

    Args:
        excludes: iterable of DType values (e.g. [DType.INT8, DType.BOOL])

    Returns:
        A set of DType values
    """
    omit = {DType.UNKNOWN, DType.UINT8, DType.UINT16, DType.SHAPE}
    omit.update(excludes if excludes else ())
    return allDTypes(excludes=omit)


def product(shape):
    value = 1
    for n in shape:
        value *= n
    return value


def get_accum_dtype_from_tgTypes(dtypes):
    # Get accumulate data-type from the test generator's defined types
    assert isinstance(dtypes, list) or isinstance(dtypes, tuple)
    return dtypes[-1]


def get_wrong_output_type(op_name, rng, input_dtype):
    if op_name == "fully_connected" or op_name == "matmul":
        if input_dtype == DType.INT8:
            incorrect_types = (
                DType.INT4,
                DType.INT8,
                DType.INT16,
                DType.INT48,
                DType.FP32,
                DType.FP16,
            )
        elif input_dtype == DType.INT16:
            incorrect_types = (
                DType.INT4,
                DType.INT8,
                DType.INT16,
                DType.INT32,
                DType.FP32,
                DType.FP16,
            )
        elif (
            input_dtype == DType.FP32
            or input_dtype == DType.FP16
            or input_dtype == DType.BF16
        ):
            incorrect_types = (
                DType.INT4,
                DType.INT8,
                DType.INT16,
                DType.INT32,
                DType.INT48,
            )
    else:
        # Assume all types but the input type are incorrect
        incorrect_types = list(usableDTypes(excludes=(input_dtype,)))
    return rng.choice(a=incorrect_types)


def get_rank_mismatch_shape(rng, output_shape):
    """
    Extends the rank of the provided output_shape by
    an arbitrary amount but ensures the total element
    count remains the same.
    """
    rank_modifier = rng.choice([1, 2, 3])
    output_shape += [1] * rank_modifier
    return output_shape


def float32_is_valid_bfloat16(f):
    """Return True if float value is valid bfloat16."""
    f32_bits = get_float32_bitstring(f)
    return f32_bits[16:] == "0" * 16


def get_float32_bitstring(f):
    """Return a big-endian string of bits representing a 32 bit float."""
    f32_bits_as_int = struct.unpack(">L", struct.pack(">f", f))[0]
    return f"{f32_bits_as_int:032b}"


def float32_to_bfloat16(f):
    """Turns fp32 value into bfloat16 by flooring.

    Floors the least significant 16 bits of the input
    fp32 value and returns this valid bfloat16 representation as fp32.
    For simplicity during bit-wrangling, ignores underlying system
    endianness and interprets as big-endian.
    Returns a bf16-valid float following system's native byte order.
    """
    f32_bits = get_float32_bitstring(f)
    f32_floored_bits = f32_bits[:16] + "0" * 16

    # Assume sys.byteorder matches system's underlying float byteorder
    fp_bytes = int(f32_floored_bits, 2).to_bytes(4, byteorder=sys.byteorder)
    return struct.unpack("@f", fp_bytes)[0]  # native byteorder


vect_f32_to_bf16 = np.vectorize(
    float32_to_bfloat16, otypes=(np.float32,)
)  # NumPy vectorize: applies function to vector faster than looping

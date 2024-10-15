# Copyright (c) 2021-2024, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
import struct
from enum import IntEnum

from tosa.DType import DType
from tosa.NanPropagationMode import NanPropagationMode
from tosa.Op import Op

# Maximum dimension size for output and inputs for RESIZE
MAX_RESIZE_DIMENSION = 16384

# Maximum rank of tensor supported by test generator.
MAX_TENSOR_RANK = 6

# Data type information dictionary
# - str: filename abbreviation
# - width: number of bytes needed for type
# - fullset: precalculated number of possible values in the data type's range, equal to 2^width
# - json: JSON type string
DTYPE_ATTRIBUTES = {
    DType.BOOL: {"str": "b", "width": 1, "fullset": 2, "json": "BOOL"},
    DType.INT4: {"str": "i4", "width": 4, "fullset": 16, "json": "INT4"},
    DType.INT8: {"str": "i8", "width": 8, "fullset": 256, "json": "INT8"},
    DType.UINT8: {"str": "u8", "width": 8, "fullset": 256, "json": "UINT8"},
    DType.INT16: {"str": "i16", "width": 16, "fullset": 65536, "json": "INT16"},
    DType.UINT16: {"str": "u16", "width": 16, "fullset": 65536, "json": "UINT16"},
    DType.INT32: {"str": "i32", "width": 32, "fullset": 1 << 32, "json": "INT32"},
    DType.INT48: {"str": "i48", "width": 48, "fullset": 1 << 48, "json": "INT48"},
    DType.SHAPE: {"str": "s", "width": 64, "fullset": 1 << 64, "json": "SHAPE"},
    DType.FP16: {"str": "f16", "width": 16, "fullset": 65536, "json": "FP16"},
    DType.BF16: {"str": "bf16", "width": 16, "fullset": 65536, "json": "BF16"},
    DType.FP32: {"str": "f32", "width": 32, "fullset": 1 << 32, "json": "FP32"},
    DType.FP8E4M3: {"str": "f8e4m3", "width": 8, "fullset": 256, "json": "FP8E4M3"},
    DType.FP8E5M2: {"str": "f8e5m2", "width": 8, "fullset": 256, "json": "FP8E5M2"},
}


class ComplianceMode(IntEnum):
    """Compliance mode types."""

    EXACT = 0
    DOT_PRODUCT = 1
    ULP = 2
    FP_SPECIAL = 3
    REDUCE_PRODUCT = 4
    ABS_ERROR = 5
    RELATIVE = 6


class DataGenType(IntEnum):
    """Data generator types."""

    PSEUDO_RANDOM = 0
    DOT_PRODUCT = 1
    FULL_RANGE = 2
    SPECIAL = 3
    FIXED_DATA = 4


class SpecialTestSet(IntEnum):
    """Special test values for SPECIAL tests"""

    DEFAULT = 0
    CAST_FP_TO_INT = 1


def dtypeWidth(dtype):
    """Get the datatype width for data types"""
    if dtype in DTYPE_ATTRIBUTES:
        return DTYPE_ATTRIBUTES[dtype]["width"]
    else:
        raise Exception(f"Unknown dtype, cannot determine width: {dtype}")


def dtypeIsFloat(dtype):
    """Is floating point data type"""
    return dtype in (DType.BF16, DType.FP16, DType.FP32, DType.FP8E4M3, DType.FP8E5M2)


def dtypeIsSupportedByCompliance(dtype):
    """Types supported by the C++ verification library."""
    if isinstance(dtype, list) or isinstance(dtype, tuple):
        dtype = dtype[0]
    return dtype in (
        DType.INT32,
        DType.INT16,
        DType.INT48,
        DType.UINT16,
        DType.UINT8,
        DType.INT4,
        DType.BOOL,
        DType.INT8,
        DType.FP32,
        DType.FP16,
        DType.BF16,
        DType.FP8E4M3,
        DType.FP8E5M2,
        DType.SHAPE,
    )


def dtypeIsSupportedByDataGen(dtype):
    """Types supported by the C++ data generation library"""
    supported_types = (
        DType.INT32,
        DType.INT16,
        DType.INT8,
        DType.FP32,
        DType.FP16,
        DType.BF16,
        DType.FP8E4M3,
        DType.FP8E5M2,
        DType.SHAPE,
    )
    if isinstance(dtype, list) or isinstance(dtype, tuple):
        return all(dt in supported_types for dt in dtype)

    return dtype in supported_types


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


def get_conv_accum_dtypes_from_tgTypes(dtypes):
    # Get accumulate data-types from the test generator's defined types
    assert isinstance(dtypes, list) or isinstance(dtypes, tuple)
    input_dtype = dtypes[0]
    output_dtype = dtypes[-1]
    # by default, accum_dtypes contains only output_dtype
    accum_dtypes = [output_dtype]
    if input_dtype == DType.FP16 and output_dtype == DType.FP16:
        accum_dtypes = [DType.FP16, DType.FP32]
    elif output_dtype == DType.BF16:
        accum_dtypes = [DType.FP32]
    return accum_dtypes


def get_wrong_output_type(op_name, rng, input_dtype):
    if op_name == "matmul":
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
        elif input_dtype == DType.FP8E4M3 or input_dtype == DType.FP8E5M2:
            incorrect_types = (
                DType.INT4,
                DType.INT8,
                DType.INT16,
                DType.INT32,
                DType.INT48,
                DType.FP32,
                DType.BF16,
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


def float32_is_valid_float8(f):
    """Return True if float value is valid float8."""
    f32_bits = get_float32_bitstring(f)
    return f32_bits[8:] == "0" * 24


def get_float32_bitstring(f):
    """Return a big-endian string of bits representing a 32 bit float."""
    f32_bits_as_int = struct.unpack(">L", struct.pack(">f", f))[0]
    return f"{f32_bits_as_int:032b}"


def normal_frac(dtype):
    if dtype == DType.FP32:
        return 23
    elif dtype == DType.FP16:
        return 10
    elif dtype == DType.BF16:
        return 7
    elif dtype == DType.FP8E4M3:
        return 3
    elif dtype == DType.FP8E5M2:
        return 2
    else:
        raise Exception(f"Unknown support dtype for normal_frac: {dtype}")


def has_nan_mode_by_enum(op_enum: int) -> bool:
    # The list of ops supporting NaN propagation mode.
    if op_enum in [
        Op.REDUCE_MAX,
        Op.REDUCE_MIN,
        Op.CLAMP,
        Op.MAXIMUM,
        Op.MINIMUM,
        Op.ARGMAX,
        Op.MAX_POOL2D,
    ]:
        return True
    return False


def has_nan_mode_by_name(op_name: str) -> bool:
    # The list of ops supporting NaN propagation mode.
    if op_name in [
        "reduce_max",
        "reduce_min",
        "clamp",
        "maximum",
        "minimum",
        "argmax",
        "max_pool2d",
    ]:
        return True
    return False


def get_nan_node(args_dict) -> NanPropagationMode:
    # Enable NaN propagation mode by default.
    if "nan_mode" not in args_dict:
        return NanPropagationMode.PROPAGATE
    nan_mode = args_dict["nan_mode"]
    # When the NaN mode is set, must be valid mode.
    assert (
        nan_mode == NanPropagationMode.PROPAGATE
        or nan_mode == NanPropagationMode.IGNORE
    ), "Invalid NaN mode"
    return nan_mode

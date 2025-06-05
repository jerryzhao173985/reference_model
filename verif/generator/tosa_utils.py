# Copyright (c) 2021-2025, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
import struct
from enum import IntEnum

import ml_dtypes
import numpy as np
from conformance.tosa_profiles import TosaProfiles
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
# - json: JSON type string - should match DType name
DTYPE_ATTRIBUTES = {
    DType.BOOL: {"str": "b", "width": 1, "fullset": 2, "json": "BOOL"},
    DType.INT4: {"str": "i4", "width": 4, "fullset": 16, "json": "INT4"},
    DType.INT8: {"str": "i8", "width": 8, "fullset": 256, "json": "INT8"},
    DType.INT16: {"str": "i16", "width": 16, "fullset": 65536, "json": "INT16"},
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
    """Compliance mode types - matching TOSA verify library supported modes."""

    EXACT = 0
    DOT_PRODUCT = 1
    ULP = 2
    FP_SPECIAL = 3
    REDUCE_PRODUCT = 4
    ABS_ERROR = 5
    RELATIVE = 6
    RESCALE_INEXACT = 7


class TestDataType(IntEnum):
    """Internal types of Test Data to test."""

    PSEUDO_RANDOM = 0
    DOT_PRODUCT = 1
    FULL_RANGE = 2
    SPECIAL = 3
    FIXED_DATA = 4
    DYNAMIC_CTC = 5


class DataGenType(IntEnum):
    """Data generator types - matching TOSA generator library supported types."""

    PSEUDO_RANDOM = 0
    DOT_PRODUCT = 1
    FULL_RANGE = 2
    SPECIAL = 3
    FIXED_DATA = 4


TESTDATA_TO_DATAGEN_TYPE = {
    TestDataType.PSEUDO_RANDOM: DataGenType.PSEUDO_RANDOM,
    TestDataType.DOT_PRODUCT: DataGenType.DOT_PRODUCT,
    TestDataType.FULL_RANGE: DataGenType.FULL_RANGE,
    TestDataType.SPECIAL: DataGenType.SPECIAL,
    TestDataType.FIXED_DATA: DataGenType.FIXED_DATA,
}


class SpecialTestSet(IntEnum):
    """Special test values for SPECIAL tests - supported by TOSA generator library."""

    DEFAULT = 0
    CAST_FP_TO_INT = 1
    ALL_MAX_VALUES = 2
    ALL_LOWEST_VALUES = 3
    ALL_ZEROES = 4
    ALL_SMALL_VALUES = 5
    FIRST_MAX_THEN_ZEROES = 6
    FIRST_LOWEST_THEN_ZEROES = 7
    FIRST_MAX_THEN_MINUS_ONES = 8
    FIRST_LOWEST_THEN_PLUS_ONES = 9


def isSpecialTest(td_type):
    """Test data type check for a special test type."""
    return td_type not in (TestDataType.PSEUDO_RANDOM, TestDataType.DOT_PRODUCT)


def dtypeWidth(dtype):
    """Get the datatype width for data types"""
    if dtype in DTYPE_ATTRIBUTES:
        return DTYPE_ATTRIBUTES[dtype]["width"]
    else:
        raise Exception(f"Unknown dtype, cannot determine width: {dtype}")


def dtypeIsFloat(dtype):
    """Is floating point data type"""
    return dtype in (DType.BF16, DType.FP16, DType.FP32, DType.FP8E4M3, DType.FP8E5M2)


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

    Excludes uncommon types (DType.UNKNOWN) in
    addition to the excludes specified by the caller, as the serializer lib
    does not support them.
    If you wish to include 'UNKNOWN', 'UINT8' or 'UINT16' use allDTypes
    instead.

    Args:
        excludes: iterable of DType values (e.g. [DType.INT8, DType.BOOL])

    Returns:
        A set of DType values
    """
    omit = {DType.UNKNOWN, DType.SHAPE}
    omit.update(excludes if excludes else ())
    return allDTypes(excludes=omit)


def product(shape):
    # Make sure we use a large enough storage type
    value = np.int64(1)
    for n in shape:
        value *= n
    return value


def get_conv_accum_dtypes_from_tgTypes(dtypes):
    # Get accumulate data-types from the test generator's defined types
    assert isinstance(dtypes, list) or isinstance(dtypes, tuple)
    input_dtype = dtypes[0]
    output_dtype = dtypes[2]

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
                DType.BF16,
            )
    else:
        # Assume all types but the input type are incorrect
        # And exclude FP8E5M2 as this is treated differently by the serializer
        excludes = (input_dtype, DType.FP8E5M2)
        incorrect_types = list(usableDTypes(excludes=excludes))
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
    """Returns the number of bits in the mantissa for a dtype."""
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
        raise ValueError(f"Unknown dtype for normal_frac: {dtype}")


def normal_min(dtype):
    """Returns the smallest normal number representable in a dtype."""
    if dtype == DType.FP32:
        return np.finfo(np.float32).smallest_normal
    elif dtype == DType.FP16:
        return np.finfo(np.float16).smallest_normal
    elif dtype == DType.BF16:
        return ml_dtypes.finfo(ml_dtypes.bfloat16).smallest_normal
    elif dtype == DType.FP8E4M3:
        return ml_dtypes.finfo(ml_dtypes.float8_e4m3fn).smallest_normal
    elif dtype == DType.FP8E5M2:
        return ml_dtypes.finfo(ml_dtypes.float8_e5m2).smallest_normal
    else:
        raise ValueError(f"Unknown dtype for normal_min: {dtype}")


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


def get_proext_from_types(dtypes, profile_extension_types_lookup=None):
    """
    Work out profile and extension(s) from input dtypes.

    Used for both test selection (see isSupported in test_select) and
    for populating the "profile" list in desc.json

    Optionally takes a dictionary usually from the TOSA_OP_LIST entry
    "profile_extension_types" that can add to the defaults.

    Returns set of profiles_supported and set of extensions_required

    NOTE: This will not work for every op, such as CAST which will
    need to work out their own profiles/extensions per test
    """
    profiles_supported = set()
    extensions_required = set()

    dtypesList = dtypes if isinstance(dtypes, (list, tuple)) else [dtypes]

    profiles_amended = False

    # Special case table supplied - add to defaults
    if profile_extension_types_lookup:
        profile_lookup = profile_extension_types_lookup.get("profile", {})
        extension_lookup = profile_extension_types_lookup.get("extension", {})
        for d in dtypesList:
            if d in profile_lookup:
                profiles_supported.update(profile_lookup[d])
                profiles_amended = True
            if d in extension_lookup:
                extensions_required.update(extension_lookup[d])

    if DType.INT4 in dtypesList:
        extensions_required.add(TosaProfiles.TosaExtInt4)
        profiles_supported.add(TosaProfiles.TosaProINT)
    if DType.INT48 in dtypesList:
        extensions_required.add(TosaProfiles.TosaExtInt16)
        profiles_supported.add(TosaProfiles.TosaProINT)
    if DType.BF16 in dtypesList:
        extensions_required.add(TosaProfiles.TosaExtBF16)
        profiles_supported.add(TosaProfiles.TosaProFP)
    if DType.FP8E5M2 in dtypesList:
        extensions_required.add(TosaProfiles.TosaExtFP8E5M2)
        profiles_supported.add(TosaProfiles.TosaProFP)
    if DType.FP8E4M3 in dtypesList:
        extensions_required.add(TosaProfiles.TosaExtFP8E4M3)
        profiles_supported.add(TosaProfiles.TosaProFP)

    if any([d in dtypesList for d in (DType.FP16, DType.FP32)]):
        profiles_supported.add(TosaProfiles.TosaProFP)
    if any(
        [
            d in dtypesList
            for d in (DType.BOOL, DType.INT4, DType.INT8, DType.INT16, DType.INT32)
        ]
    ):
        profiles_supported.add(TosaProfiles.TosaProINT)

    # We are not expecting to support multiple different profiles unless
    # overridden by the profile_extension_types_lookup
    assert (
        profiles_amended or len(profiles_supported) == 1
    ), f"Mixed types, not sure which profile for: {[DTYPE_ATTRIBUTES[d]['str'] for d in dtypesList]}"

    # But we are expecting at least 1!
    assert (
        profiles_supported
    ), f"Failed to determine profile or extension from: {[DTYPE_ATTRIBUTES[d]['str'] for d in dtypesList]}"

    return profiles_supported, extensions_required

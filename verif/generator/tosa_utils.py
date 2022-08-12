# Copyright (c) 2021-2022, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
from tosa.DType import DType

# Maximum dimension size for output and inputs for RESIZE
MAX_RESIZE_DIMENSION = 16384


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
    omit = {DType.UNKNOWN, DType.UINT8, DType.UINT16}
    omit.update(excludes if excludes else ())
    return allDTypes(excludes=omit)


def product(shape):
    value = 1
    for n in shape:
        value *= n
    return value


def get_accum_dtype_from_tgTypes(dtypes):
    # Get accumulate data-type from the test generator's defined types
    if isinstance(dtypes, list) or isinstance(dtypes, tuple):
        return dtypes[-1]
    else:
        return dtypes


def get_wrong_output_type(op_name, rng, input_dtype):
    if op_name == "fully_connected" or op_name == "matmul":
        if input_dtype == DType.INT8:
            incorrect_types = (
                DType.INT4,
                DType.INT8,
                DType.INT16,
                DType.INT48,
                DType.FLOAT,
                DType.FP16,
            )
        elif input_dtype == DType.INT16:
            incorrect_types = (
                DType.INT4,
                DType.INT8,
                DType.INT16,
                DType.INT32,
                DType.FLOAT,
                DType.FP16,
            )
        elif input_dtype == DType.FLOAT or input_dtype == DType.FP16:
            incorrect_types = (
                DType.INT4,
                DType.INT8,
                DType.INT16,
                DType.INT32,
                DType.INT48,
            )
    return rng.choice(a=incorrect_types)

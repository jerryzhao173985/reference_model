"""TOSA result checker script."""
# Copyright (c) 2020-2022, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
import argparse
from enum import Enum
from enum import IntEnum
from enum import unique
from pathlib import Path

import numpy as np
from generator.tosa_utils import float32_is_valid_bfloat16

##################################
color_printing = True


@unique
class LogColors(Enum):
    """Shell escape sequence colors for logging."""

    NONE = "\u001b[0m"
    GREEN = "\u001b[32;1m"
    RED = "\u001b[31;1m"
    YELLOW = "\u001b[33;1m"
    BOLD_WHITE = "\u001b[1m"


def set_print_in_color(enabled):
    """Set color printing to enabled or disabled."""
    global color_printing
    color_printing = enabled


def print_color(color, msg):
    """Print color status messages if enabled."""
    global color_printing
    if not color_printing:
        print(msg)
    else:
        print("{}{}{}".format(color.value, msg, LogColors.NONE.value))


@unique
class TestResult(IntEnum):
    """Test result values."""

    # Note: PASS must be 0 for command line return success
    PASS = 0
    MISSING_FILE = 1
    INCORRECT_FORMAT = 2
    MISMATCH = 3
    INTERNAL_ERROR = 4


TestResultErrorStr = [
    "",
    "Missing file",
    "Incorrect format",
    "Mismatch",
    "Internal error",
]
##################################

DEFAULT_FP_TOLERANCE = 1e-3


def test_check(
    reference_path,
    result_path,
    test_name="test",
    quantize_tolerance=0,
    float_tolerance=DEFAULT_FP_TOLERANCE,
    misc_checks=[],
):
    """Check if the result is the same as the expected reference."""
    if not reference_path.is_file():
        print_color(LogColors.RED, "Reference MISSING FILE {}".format(test_name))
        msg = "Missing reference file: {}".format(reference_path)
        return (TestResult.MISSING_FILE, 0.0, msg)
    if not result_path.is_file():
        print_color(LogColors.RED, "Results MISSING FILE {}".format(test_name))
        msg = "Missing result file: {}".format(result_path)
        return (TestResult.MISSING_FILE, 0.0, msg)

    try:
        test_result = np.load(result_path)
    except Exception as e:
        print_color(LogColors.RED, "Results INCORRECT FORMAT {}".format(test_name))
        msg = "Incorrect numpy format of {}\nnumpy.load exception: {}".format(
            result_path, e
        )
        return (TestResult.INCORRECT_FORMAT, 0.0, msg)
    try:
        reference_result = np.load(reference_path)
    except Exception as e:
        print_color(LogColors.RED, "Reference INCORRECT FORMAT {}".format(test_name))
        msg = "Incorrect numpy format of {}\nnumpy.load exception: {}".format(
            reference_path, e
        )
        return (TestResult.INCORRECT_FORMAT, 0.0, msg)

    # Type comparison
    if test_result.dtype != reference_result.dtype:
        print_color(LogColors.RED, "Results TYPE MISMATCH {}".format(test_name))
        msg = "Mismatch results type: Expected {}, got {}".format(
            reference_result.dtype, test_result.dtype
        )
        return (TestResult.MISMATCH, 0.0, msg)

    # Size comparison
    # Size = 1 tensors can be equivalently represented as having rank 0 or rank
    # >= 0, allow that special case
    test_result = np.squeeze(test_result)
    reference_result = np.squeeze(reference_result)
    difference = None

    if np.shape(test_result) != np.shape(reference_result):
        print_color(LogColors.RED, "Results MISCOMPARE {}".format(test_name))
        msg = "Shapes mismatch: Reference {} vs {}".format(
            np.shape(test_result), np.shape(reference_result)
        )
        return (TestResult.MISMATCH, 0.0, msg)

    # Perform miscellaneous checks
    if "bf16" in misc_checks:
        # Ensure floats are valid bfloat16 values
        test_res_is_bf16 = all([float32_is_valid_bfloat16(f) for f in test_result.flat])
        ref_res_is_bf16 = all(
            [float32_is_valid_bfloat16(f) for f in reference_result.flat]
        )
        if not (test_res_is_bf16 and ref_res_is_bf16):
            msg = (
                "All output values must be valid bfloat16. "
                "reference_result: {ref_res_is_bf16}; test_result: {test_res_is_bf16}"
            )
            return (TestResult.INCORRECT_FORMAT, 0.0, msg)

    # for quantized test, allow +-(quantize_tolerance) error
    if reference_result.dtype == np.int32 or reference_result.dtype == np.int64:

        if np.all(np.absolute(reference_result - test_result) <= quantize_tolerance):
            print_color(LogColors.GREEN, "Results PASS {}".format(test_name))
            return (TestResult.PASS, 0.0, "")
        else:
            tolerance = quantize_tolerance + 1
            while not np.all(
                np.absolute(reference_result - test_result) <= quantize_tolerance
            ):
                tolerance = tolerance + 1
                if tolerance > 10:
                    break

            if tolerance > 10:
                msg = "Integer result does not match and is greater than 10 difference"
            else:
                msg = (
                    "Integer result does not match but is within {} difference".format(
                        tolerance
                    )
                )
            # Fall-through to below to add failure values
            difference = reference_result - test_result

    elif reference_result.dtype == bool:
        assert test_result.dtype == bool
        # All boolean values must match, xor will show up differences
        test = np.array_equal(reference_result, test_result)
        if np.all(test):
            print_color(LogColors.GREEN, "Results PASS {}".format(test_name))
            return (TestResult.PASS, 0.0, "")
        msg = "Boolean result does not match"
        tolerance = 0.0
        difference = None
        # Fall-through to below to add failure values

    # TODO: update for fp16 tolerance
    elif reference_result.dtype == np.float32 or reference_result.dtype == np.float16:
        tolerance = float_tolerance
        if np.allclose(reference_result, test_result, atol=tolerance, equal_nan=True):
            print_color(LogColors.GREEN, "Results PASS {}".format(test_name))
            return (TestResult.PASS, tolerance, "")
        msg = "Float result does not match within tolerance of {}".format(tolerance)
        difference = reference_result - test_result
        # Fall-through to below to add failure values
    else:
        print_color(LogColors.RED, "Results UNSUPPORTED TYPE {}".format(test_name))
        msg = "Unsupported results type: {}".format(reference_result.dtype)
        return (TestResult.MISMATCH, 0.0, msg)

    # Fall-through for mismatch failure to add values to msg
    print_color(LogColors.RED, "Results MISCOMPARE {}".format(test_name))
    np.set_printoptions(threshold=128, edgeitems=2)

    if difference is not None:
        tolerance_needed = np.amax(np.absolute(difference))
        msg = "{}\n-- tolerance_needed: {}".format(msg, tolerance_needed)

    msg = "{}\n>> reference_result: {}\n{}".format(
        msg, reference_result.shape, reference_result
    )
    msg = "{}\n<< test_result: {}\n{}".format(msg, test_result.shape, test_result)

    if difference is not None:
        msg = "{}\n!! difference_result: \n{}".format(msg, difference)
    return (TestResult.MISMATCH, tolerance, msg)


def main(argv=None):
    """Check that the supplied reference and result files have the same contents."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "reference_path", type=Path, help="the path to the reference file to test"
    )
    parser.add_argument(
        "result_path", type=Path, help="the path to the result file to test"
    )
    parser.add_argument(
        "--fp-tolerance", type=float, default=DEFAULT_FP_TOLERANCE, help="FP tolerance"
    )
    args = parser.parse_args(argv)

    result, tolerance, msg = test_check(
        args.reference_path, args.result_path, float_tolerance=args.fp_tolerance
    )
    if result != TestResult.PASS:
        print(msg)

    return result


if __name__ == "__main__":
    exit(main())

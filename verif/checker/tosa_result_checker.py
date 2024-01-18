"""TOSA result checker script."""
# Copyright (c) 2020-2024, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
import argparse
import json
from enum import IntEnum
from enum import unique
from pathlib import Path

import numpy as np
from checker.color_print import LogColors
from checker.color_print import print_color
from checker.verifier import VerifierError
from checker.verifier import VerifierLibrary
from generator.tosa_utils import float32_is_valid_bfloat16
from schemavalidation.schemavalidation import TestDescSchemaValidator


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
result_printing = True


def set_print_result(enabled):
    """Set whether to print out or not."""
    global result_printing
    result_printing = enabled


def _print_result(color, msg):
    """Print out result."""
    global result_printing
    if result_printing:
        print_color(color, msg)


def compliance_check(
    imp_result_data,
    ref_result_data,
    bnd_result_data,
    test_name,
    compliance_config,
    ofm_name,
    verify_lib_path,
):
    if verify_lib_path is None:
        error = "Please supply --verify-lib-path"
    else:
        error = None
        try:
            vlib = VerifierLibrary(verify_lib_path)
        except VerifierError as e:
            error = str(e)

    if error is not None:
        _print_result(LogColors.RED, f"INTERNAL ERROR {test_name}")
        msg = f"Could not load verfier library: {error}"
        return (TestResult.INTERNAL_ERROR, 0.0, msg)

    success = vlib.verify_data(
        ofm_name, compliance_config, imp_result_data, ref_result_data, bnd_result_data
    )
    if success:
        _print_result(LogColors.GREEN, f"Compliance Results PASS {test_name}")
        return (TestResult.PASS, 0.0, "")
    else:
        _print_result(LogColors.RED, f"Results NON-COMPLIANT {test_name}")
        return (
            TestResult.MISMATCH,
            0.0,
            f"Non-compliance results found for {ofm_name}",
        )


def test_check(
    ref_result_path,
    imp_result_path,
    test_name=None,
    quantize_tolerance=0,
    float_tolerance=DEFAULT_FP_TOLERANCE,
    misc_checks=[],
    test_desc=None,
    bnd_result_path=None,
    ofm_name=None,
    verify_lib_path=None,
):
    """Check if the result is the same as the expected reference."""
    if test_desc:
        # New compliance method - first get test details
        try:
            TestDescSchemaValidator().validate_config(test_desc)
        except Exception as e:
            _print_result(LogColors.RED, f"Test INCORRECT FORMAT {test_name}")
            msg = f"Incorrect test format: {e}"
            return (TestResult.INCORRECT_FORMAT, 0.0, msg)

    if test_name is None:
        test_name = "test"

    paths = [imp_result_path, ref_result_path, bnd_result_path]
    names = ["Implementation", "Reference", "Bounds"]
    arrays = [None, None, None]

    # Check the files exist and are in the right format
    for idx, path in enumerate(paths):
        name = names[idx]
        if path is None and name == "Bounds":
            # Bounds can be None - skip it
            continue
        if not path.is_file():
            _print_result(LogColors.RED, f"{name} MISSING FILE {test_name}")
            msg = f"Missing {name} file: {str(path)}"
            return (TestResult.MISSING_FILE, 0.0, msg)
        try:
            arrays[idx] = np.load(path)
        except Exception as e:
            _print_result(LogColors.RED, f"{name} INCORRECT FORMAT {test_name}")
            msg = f"Incorrect numpy format of {str(path)}\nnumpy.load exception: {e}"
            return (TestResult.INCORRECT_FORMAT, 0.0, msg)

    if test_desc and "meta" in test_desc and "compliance" in test_desc["meta"]:
        # Switch to using the verifier library for full compliance
        if ofm_name is None:
            ofm_name = test_desc["ofm_name"][0]
            if len(test_desc["ofm_name"]) > 1:
                _print_result(LogColors.RED, f"Output Name MISSING FILE {test_name}")
                msg = "Must specify output name (ofm_name) to check as multiple found in desc.json"
                return (TestResult.MISSING_FILE, 0.0, msg)

        compliance_json = test_desc["meta"]["compliance"]

        return compliance_check(
            *arrays,
            test_name,
            compliance_json,
            ofm_name,
            verify_lib_path,
        )

    # Else continue with original checking method
    test_result, reference_result, _ = arrays

    # Type comparison
    if test_result.dtype != reference_result.dtype:
        _print_result(LogColors.RED, "Results TYPE MISMATCH {}".format(test_name))
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
        _print_result(LogColors.RED, "Results MISCOMPARE {}".format(test_name))
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
    if reference_result.dtype in (
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
    ):

        if np.all(np.absolute(reference_result - test_result) <= quantize_tolerance):
            _print_result(LogColors.GREEN, "Results PASS {}".format(test_name))
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
            _print_result(LogColors.GREEN, "Results PASS {}".format(test_name))
            return (TestResult.PASS, 0.0, "")
        msg = "Boolean result does not match"
        tolerance = 0.0
        difference = None
        # Fall-through to below to add failure values

    # TODO: update for fp16 tolerance
    elif reference_result.dtype == np.float32 or reference_result.dtype == np.float16:
        tolerance = float_tolerance
        if np.allclose(reference_result, test_result, atol=tolerance, equal_nan=True):
            _print_result(LogColors.GREEN, "Results PASS {}".format(test_name))
            return (TestResult.PASS, tolerance, "")
        msg = "Float result does not match within tolerance of {}".format(tolerance)
        difference = reference_result - test_result
        # Fall-through to below to add failure values
    else:
        _print_result(LogColors.RED, "Results UNSUPPORTED TYPE {}".format(test_name))
        msg = "Unsupported results type: {}".format(reference_result.dtype)
        return (TestResult.MISMATCH, 0.0, msg)

    # Fall-through for mismatch failure to add values to msg
    _print_result(LogColors.RED, "Results MISCOMPARE {}".format(test_name))
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
        "ref_result_path",
        type=Path,
        help="path to the reference model result file to check",
    )
    parser.add_argument(
        "imp_result_path",
        type=Path,
        help="path to the implementation result file to check",
    )
    parser.add_argument(
        "--fp-tolerance", type=float, default=DEFAULT_FP_TOLERANCE, help="FP tolerance"
    )
    parser.add_argument(
        "--test-path", type=Path, help="path to the test that produced the results"
    )
    # Deprecate the incorrectly formatted option by hiding it
    parser.add_argument("--test_path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--bnd-result-path",
        type=Path,
        help="path to the reference model bounds result file for the dot product compliance check",
    )
    parser.add_argument(
        "--ofm-name",
        type=str,
        help="name of the output tensor to check, defaults to the first ofm_name listed in the test",
    )
    parser.add_argument(
        "--verify-lib-path",
        type=Path,
        help="path to TOSA verify library",
    )
    args = parser.parse_args(argv)

    if args.test_path:
        # Get details from the test path
        test_desc_path = args.test_path / "desc.json"
        if not args.test_path.is_dir() or not test_desc_path.is_file():
            print(f"Invalid test directory {str(args.test_path)}")
            return TestResult.MISSING_FILE

        try:
            with test_desc_path.open("r") as fd:
                test_desc = json.load(fd)
        except Exception as e:
            print(f"Invalid test description file {str(test_desc_path)}: {e}")
            return TestResult.INCORRECT_FORMAT
        test_name = args.test_path.name
    else:
        test_desc = None
        test_name = None

    result, tolerance, msg = test_check(
        args.ref_result_path,
        args.imp_result_path,
        float_tolerance=args.fp_tolerance,
        test_name=test_name,
        test_desc=test_desc,
        bnd_result_path=args.bnd_result_path,
        ofm_name=args.ofm_name,
        verify_lib_path=args.verify_lib_path,
    )
    if result != TestResult.PASS:
        print(msg)

    return result


if __name__ == "__main__":
    exit(main())

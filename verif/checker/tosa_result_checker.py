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
from runner.run_command import run_sh_command
from runner.run_command import RunShCommandError
from schemavalidation.schemavalidation import TestDescSchemaValidator


@unique
class TosaVerifyReturnCode(IntEnum):
    """The tosa_verify exit codes."""

    TOSA_COMPLIANT = 0
    TOSA_ERROR = 1
    TOSA_NONCOMPLIANT = 2


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
        return (TestResult.INTERNAL_ERROR, msg)

    success = vlib.verify_data(
        ofm_name, compliance_config, imp_result_data, ref_result_data, bnd_result_data
    )
    if success:
        _print_result(LogColors.GREEN, f"Compliance Results PASS {test_name}")
        return (TestResult.PASS, "")
    else:
        _print_result(LogColors.RED, f"Results NON-COMPLIANT {test_name}")
        return (
            TestResult.MISMATCH,
            f"Non-compliance results found for {ofm_name}",
        )


def tosa_verify(
    imp_result_path,
    ref_result_path,
    bnd_result_path,
    test_desc_path,
    ofm_name,
    verify_path,
):
    # Where ever the test description resides will be the test name
    test_name = test_desc_path.parent.name

    if verify_path is None or not verify_path.is_file():
        error = "Please supply valid --verify-path"
        result = TosaVerifyReturnCode.TOSA_ERROR
    else:
        cmd = [
            str(verify_path),
            "--test_desc",
            str(test_desc_path),
            "--imp_result_file",
            str(imp_result_path),
            "--ref_result_file",
            str(ref_result_path),
        ]

        if ofm_name:
            cmd.extend(["--ofm_name", ofm_name])

        if bnd_result_path:
            cmd.extend(["--bnd_result_file", str(bnd_result_path)])

        try:
            run_sh_command(cmd, True, capture_output=True)
            result = TosaVerifyReturnCode.TOSA_COMPLIANT
        except (RunShCommandError, PermissionError) as e:
            error = e.stderr
            result = e.return_code

    if result == TosaVerifyReturnCode.TOSA_NONCOMPLIANT:
        _print_result(LogColors.RED, f"Results NON-COMPLIANT {test_name}")
        name = ofm_name if ofm_name else test_name
        return (
            TestResult.MISMATCH,
            f"Non-compliance results found for {name}",
        )
    elif result != TosaVerifyReturnCode.TOSA_COMPLIANT:
        _print_result(LogColors.RED, f"INTERNAL ERROR {test_name}")
        msg = f"Error during verification: {error}"
        return (TestResult.INTERNAL_ERROR, msg)
    else:
        _print_result(LogColors.GREEN, f"Compliance Results PASS {test_name}")
        return (TestResult.PASS, "")


def parse_args(argv):
    """Parse command line."""
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
        "--test-path",
        type=Path,
        help="path to the test that produced the results",
        required=True,
    )

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

    verify_group = parser.add_mutually_exclusive_group(required=True)
    verify_group.add_argument(
        "--verify-path",
        type=Path,
        help="path to TOSA verify executable",
    )

    # Deprecated old options by hiding then
    parser.add_argument("--fp-tolerance", type=float, help=argparse.SUPPRESS)
    verify_group.add_argument("--verify-lib-path", type=Path, help=argparse.SUPPRESS)

    return parser.parse_args(argv)


def main(argv=None):
    """Check that the supplied reference and result files have the same contents."""

    print_color(
        LogColors.YELLOW, "DEPRECATION NOTICE: Please use `tosa_verify` instead."
    )

    args = parse_args(argv)

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

    # Check test description
    try:
        TestDescSchemaValidator().validate_config(test_desc)
        if "meta" not in test_desc or "compliance" not in test_desc["meta"]:
            raise KeyError("Missing compliance meta data")
    except Exception as e:
        _print_result(LogColors.RED, f"Test INCORRECT FORMAT {test_name}")
        print(f"Incorrect test description format: {e}")
        return TestResult.INCORRECT_FORMAT

    if args.verify_path:
        # Default is to use the tosa_verify executable
        result, msg = tosa_verify(
            args.imp_result_path,
            args.ref_result_path,
            args.bnd_result_path,
            test_desc_path,
            args.ofm_name,
            args.verify_path,
        )
    else:
        # Backwards compatibility support using the verifier library
        ofm_name = args.ofm_name
        if ofm_name is None:
            ofm_name = test_desc["ofm_name"][0]
            if len(test_desc["ofm_name"]) > 1:
                _print_result(LogColors.RED, f"Output Name MISSING FILE {test_name}")
                print(
                    "Must specify output name (--ofm-name) as multiple outputs found in test description"
                )
                return TestResult.MISSING_FILE

        paths = [args.imp_result_path, args.ref_result_path, args.bnd_result_path]
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
                print(f"Missing {name} file: {str(path)}")
                return TestResult.MISSING_FILE
            try:
                arrays[idx] = np.load(path)
            except Exception as e:
                _print_result(LogColors.RED, f"{name} INCORRECT FORMAT {test_name}")
                print(
                    f"Incorrect numpy format of {str(path)}\nnumpy.load exception: {e}"
                )
                return TestResult.INCORRECT_FORMAT

        compliance_json = test_desc["meta"]["compliance"]
        result, msg = compliance_check(
            *arrays, test_name, compliance_json, ofm_name, args.verify_lib_path
        )

    if result != TestResult.PASS:
        print(msg)

    return result


if __name__ == "__main__":
    exit(main())

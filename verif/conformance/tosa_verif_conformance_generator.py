#!/usr/bin/env python3
# Copyright (c) 2021-2022, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
"""Build conformance tests.

Steps:
- Specific input shapes (or tests) are specified and produced by using the
  settings in the .json files.
- Tests are selected to produce a good coverage.
- Tests are run on the reference model to produce the correct output files.
- Tests are converted into JSON format and saved to desired output directory.
"""
import argparse
import json
import logging
import multiprocessing as mp
import os
import shlex
import shutil
import subprocess
from functools import partial
from itertools import tee
from pathlib import Path

from conformance.test_select import Operator
from convert2conformance.convert2conformance import main as c2c_main
from distutils.dir_util import copy_tree

logging.basicConfig()
logger = logging.getLogger("tosa_verif_conformance_generator")

# Configuration for each TOSA profile
PROFILE_OPS_INFO = {
    "tosa-bi": {
        "operator_test_params": "tosa_base_profile_ops_info.json",
        "framework_tests": "tosa_base_profile_framework_ops_info.json",
        "exclude_types": [],
    }
}

LOCATION_REF_MODEL_BINARY = Path("build/reference_model/tosa_reference_model")

DEFAULT_SEED = 42


class GenConformanceError(Exception):
    """Generation error reporting exception."""

    pass


def _run_sh_command(args, cwd, full_cmd):
    """Run an external command and capture stdout/stderr."""
    # Quote the command line for printing
    full_cmd_esc = [shlex.quote(x) for x in full_cmd]
    if args.capture_output:
        logger.debug(f"Command: {full_cmd_esc}")

    rc = subprocess.run(
        full_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd
    )

    if args.capture_output:
        stdout = rc.stdout.decode("utf-8")
        logger.debug(f"stdout: \n{stdout}")
    if rc.returncode != 0:

        raise Exception(
            "Error running command: {}.\n{}".format(
                " ".join(full_cmd_esc), rc.stderr.decode("utf-8")
            )
        )
    return (rc.stdout, rc.stderr)


def build_op_tests(args, operator, test_params):
    """Build tests for a given operator.

    Builds a set of tests based on the parameters defined in test_params

    Returns operator output directory
    """
    assert operator in test_params

    build_tests_cmd = "tosa_verif_build_tests"
    op_build_dir = args.build_dir

    ref_cmd_base = [
        build_tests_cmd,
        "--filter",
        operator,
        "-o",
        str(op_build_dir),
        "--seed",
        str(args.random_seed),
    ]

    ref_cmds = []

    if args.test_type in ["positive", "both"]:
        # Append extra parameters and run test generator for each set of parameters.
        for arglist in test_params[operator]["generator_args"]:
            ref_cmd_pos_test = ref_cmd_base.copy()
            ref_cmd_pos_test.extend(["--test-type", "positive"])
            ref_cmd_pos_test.extend(arglist)
            ref_cmds.append(ref_cmd_pos_test)

    if args.test_type in ["negative", "both"]:
        # Get target-dtypes options only to limit tests to those needed
        target_dtypes_args = []
        for arglist in test_params[operator]["generator_args"]:
            idx = 0
            while idx < len(arglist):
                if arglist[idx] == "--target-dtype":
                    if arglist[idx + 1] not in target_dtypes_args:
                        target_dtypes_args.extend(arglist[idx : idx + 2])
                    idx += 1  # skip over option (and then argument below)
                idx += 1
        ref_cmd_neg_test = ref_cmd_base.copy()
        ref_cmd_neg_test.extend(["--test-type", "negative"])
        # Limit sizes of negative tests
        ref_cmd_neg_test.extend(["--tensor-dim-range", "1,16"])
        ref_cmd_neg_test.extend(target_dtypes_args)
        ref_cmds.append(ref_cmd_neg_test)

    logger.debug(f"Creating {operator} tests with {len(ref_cmds)} parameter(s)")
    error = False
    for i, cmd in enumerate(ref_cmds):
        try:
            _run_sh_command(args, args.ref_model_dir.absolute(), cmd)
            logger.info(
                f"{operator} test batch {(i+1)}/{len(ref_cmds)} created successfully"
            )
        except Exception as e:
            logger.error(
                f"{operator} test batch {(i+1)}/{len(ref_cmds)} unsuccessful, skipping"
            )
            logger.error(f" build_op_tests error: {e} ")
            error = True
    if error:
        raise (GenConformanceError())

    return op_build_dir


def _check_to_include_test(profile, test_name, exclude_negative_tests=False):
    """Check test name for exclusions, return False to indicate excluded."""
    excludes = ["ERRORIF"] if exclude_negative_tests else []
    excludes.extend(PROFILE_OPS_INFO[profile]["exclude_types"])

    for exclusion in excludes:
        if f"_{exclusion}_" in test_name:
            return False
    return True


def _get_all_tests_list(
    profile, test_root_dir, operator, exclude_negative_tests=False, include_all=False
):
    """Create test list based on tests in the test_dir."""
    test_dir = test_root_dir / operator
    if not test_dir.is_dir():
        # Tests are split into multiple dirs, for example: conv2d_1x1, conv2d_3x3
        test_dir = test_root_dir
        directories = [
            tdir for tdir in test_dir.glob("*") if tdir.name.startswith(operator)
        ]
    else:
        directories = [test_dir]

    tests = []
    for tdir in directories:
        tests.extend(
            [
                test
                for test in tdir.glob("*")
                if include_all
                or _check_to_include_test(profile, test.name, exclude_negative_tests)
            ]
        )
    return tests


def generate_results(args, operator, op_build_dir, tests=None):
    """Run tests on reference model and save result to the test directory."""
    num_cores = args.num_cores
    run_tests_cmd = "tosa_verif_run_tests"

    ref_model_path = args.ref_model_dir / LOCATION_REF_MODEL_BINARY
    ref_cmd_base = ref_cmd = [
        run_tests_cmd,
        "--ref-model-path",
        str(ref_model_path.absolute()),
        "-j",
        str(num_cores),
        "-v",
        "-t",
    ]
    ref_cmds = []

    if not tests:
        # Do not need to run ERRORIF tests as they don't have result files
        tests = _get_all_tests_list(
            args.profile, op_build_dir, operator, exclude_negative_tests=True
        )

    for test in tests:
        ref_cmd = ref_cmd_base.copy()
        ref_cmd.append(str(test))
        ref_cmds.append(ref_cmd)

    fail_string = "UNEXPECTED_FAILURE"
    failed_counter = 0

    job_pool = mp.Pool(args.num_cores)
    sh_partial = partial(_run_sh_command, args, args.ref_model_dir.absolute())
    pool_results = job_pool.map(sh_partial, ref_cmds)
    job_pool.close()
    job_pool.join()

    # Use captured output for run_sh_command to work out if test passed.
    for i, rc in enumerate(pool_results):
        if fail_string in str(rc[0]):
            logger.error(f"Test {i+1}/{len(ref_cmds)}: {ref_cmds[i][-1]} failed.")
            failed_counter += 1
        else:
            logger.info(f"Test {i+1}/{len(ref_cmds)}: {ref_cmds[i][-1]} passed.")

    logger.info(f"{len(ref_cmds)-failed_counter}/{len(ref_cmds)} tests passed")
    logger.info("Ran tests on model and saved results of passing tests")


def convert_tests(
    args,
    operator,
    op_build_dir,
    output_dir,
    profiles,
    tests=None,
    group=None,
    trim_op_subdir=False,
):
    """Convert tests to JSON and save to output directory."""
    ref_model_dir = args.ref_model_dir

    if group:
        output_dir = output_dir / group

    ref_cmd_base = ["--ref-model-directory", str(ref_model_dir)]
    for profile in profiles:
        ref_cmd_base.extend(["--profile", profile])
    if args.framework_schema:
        ref_cmd_base.extend(["--framework-schema", str(args.framework_schema)])
    ref_cmd_base.append("--output-directory")

    ref_cmds = []

    if not tests:
        tests = _get_all_tests_list(args.profile, op_build_dir, operator)
        logger.info(f"Converting all {args.profile} profile tests")

    # Controls if we copy the tests in their operator sub-directory or not
    output_dir_relative_pos = -1 if trim_op_subdir else -2
    for test in tests:
        logger.info(f"Test chosen: {test}")
        ref_cmd = ref_cmd_base.copy()
        full_output_directory = output_dir / test.relative_to(
            *test.parts[:output_dir_relative_pos]
        )
        ref_cmd.append(str(full_output_directory))
        ref_cmd.append(str(test))
        ref_cmds.append(ref_cmd)

    if len(ref_cmds) == 0:
        logger.warning("No tests found. Nothing to convert")
        return

    job_pool = mp.Pool(args.num_cores)

    pool_results = job_pool.map(c2c_main, ref_cmds)
    job_pool.close()
    job_pool.join()

    failed_counter = 0
    for i, result in enumerate(pool_results):
        if result != 0:
            logger.error(
                f"test {i+1}/{len(ref_cmds)}: {ref_cmds[i][-1]} failed to convert."
            )
            failed_counter += 1
        else:
            logger.info(f"test {i+1}/{len(ref_cmds)}: {ref_cmds[i][-1]} converted")
    logger.info(
        f"{len(ref_cmds)-failed_counter}/{len(ref_cmds)} tests successfully converted"
    )

    if failed_counter > 0:
        logger.error(f"Stopping due to {failed_counter} test conversion errors")
        raise (GenConformanceError())

    logger.info("Converted tests to JSON and saved to output directory")

    return output_dir


def get_op_tests_selection(args, operator, op_build_dir, test_params, negative=False):
    """Use test picker to get subsection of tests generated."""
    assert operator in test_params
    try:
        op_params = test_params[operator]
        op = Operator.registry[operator](
            op_build_dir,
            op_params,
            negative,
            exclude_types=PROFILE_OPS_INFO[args.profile]["exclude_types"],
        )
    except KeyError:
        logger.error(f"{operator} operator is not supported by test_select")
        raise (GenConformanceError())

    return op.select_tests()


def check_op_tests(args, operator, output_dir):
    """Move test folders than contain files larger than 30MB to new directory."""
    destination_dir = str(args.output_dir) + "_large_files"

    tests = _get_all_tests_list(args.profile, output_dir, operator, include_all=True)
    if not tests:
        logger.error(
            f"Couldn't find any tests to size check for {operator} in {output_dir}"
        )
        raise (GenConformanceError())

    for tdir in tests:
        move_dir = False
        test_files = [file for file in tdir.glob("*")]
        for file in test_files:
            file_size = os.stat(file).st_size / 1024**2
            if file_size > 30:
                move_dir = True

        if move_dir:
            move_destination = destination_dir / tdir.relative_to(output_dir)
            logger.warning(
                f"{tdir.relative_to(output_dir)} contains files that are too large (>30MB), test moved to new folder: {destination_dir}"
            )

            if move_destination.is_dir():
                logger.warning(
                    f"{move_destination} directory already exists, deleting existing."
                )
                shutil.rmtree(str(move_destination))
            shutil.move(str(tdir), move_destination)


def copy_rename_framework_tests(args, operator, test_picks):
    """Copy framework tests into new folder and rename them if needed.

    The tests are renamed to match the framework operator names if an
    alternate name has been used instead.
    """
    framework_tests_dir = args.framework_tests_dir
    new_tests_dir = args.build_dir / "frameworks" / operator
    os.makedirs(new_tests_dir, exist_ok=True)

    # Get the framework tests operator name
    if "alternate_names" in test_picks[operator]:
        alternate_names = test_picks[operator]["alternate_names"]
    else:
        alternate_names = [operator]

    # Get the alternate named test directories for the operator
    for alt_name in alternate_names:
        test_prefix = f"test_{alt_name}"
        test_dirs = list(framework_tests_dir.glob(f"{test_prefix}_*"))

        # Copy tests to new directory and rename to match framework operator names
        # - if there is just 1 alternate name, replace the full test prefix
        #       test_add_... -> add_...
        # - if there are multiple alternate names, just replace the "test"
        #       test_concatv2_... -> concatenation_concatv2_...
        old_prefix = test_prefix if len(alternate_names) == 1 else "test"

        for tdir in test_dirs:
            new_test_name = tdir.name.replace(old_prefix, operator)
            copy_destination = new_tests_dir / new_test_name
            logger.debug(f"copying test folder {tdir} to {copy_destination}")
            copy_tree(str(tdir), str(copy_destination))

    logger.info(f"Copied and renamed {len(test_dirs)} framework test folders")
    return new_tests_dir.parent


def get_framework_tests_selection(args, operator, test_picks, op_build_dir):
    """Get the list of pre-chosen tests with relative paths."""
    try:
        tests = test_picks[operator]["tests"]
    except KeyError:
        logger.error(f"Framework test selection not defined for {operator} operator")
        raise (GenConformanceError())

    test_paths = [op_build_dir / operator / test for test in tests]
    return test_paths


def parse_args(argv=None):
    """Parse the arguments."""
    parser = argparse.ArgumentParser()
    profiles = list(PROFILE_OPS_INFO.keys())
    parser.add_argument(
        "--profile",
        dest="profile",
        choices=profiles,
        default=profiles[0],
        type=str,
        help=f"TOSA profile (default is {profiles[0]})",
    )
    parser.add_argument(
        "--operators",
        type=str,
        nargs="*",
        help="The operator(s) to create tests for, if not supplied all tests will be created",
    )
    parser.add_argument(
        "--unit-tests",
        dest="unit_tests",
        choices=["operator", "framework", "both"],
        default="operator",
        type=str,
        help="Which unit tests are produced (default is operator)",
    )
    parser.add_argument(
        "--test-type",
        dest="test_type",
        choices=["positive", "negative", "both"],
        default="both",
        type=str,
        help="Type of tests produced (default is both)",
    )
    parser.add_argument(
        "--ref-model-directory",
        dest="ref_model_dir",
        type=Path,
        required=True,
        help="Reference Model directory (must be pre-built)",
    )
    parser.add_argument(
        "--seed",
        dest="random_seed",
        default=DEFAULT_SEED,
        type=int,
        help="Random test seed",
    )
    parser.add_argument(
        "--framework-tests-directory",
        dest="framework_tests_dir",
        type=Path,
        default=Path.cwd() / "tests",
        help="The pre-built framework tests directory (default is tests)",
    )
    parser.add_argument(
        "--framework-schema",
        dest="framework_schema",
        type=Path,
        help="Framework flatbuffers schema needed to convert framework models",
    )
    parser.add_argument(
        "--build-directory",
        dest="build_dir",
        type=Path,
        default=Path.cwd() / "conformance_build",
        help="Temporary build directory for files created during this process (default is conformance_build)",
    )
    parser.add_argument(
        "--output-directory",
        dest="output_dir",
        type=Path,
        default=Path.cwd() / "conformance",
        help="Output directory (default is conformance)",
    )
    script_dir = Path(__file__).parent.absolute()
    parser.add_argument(
        "--test-param-json-directory",
        dest="param_json_dir",
        type=Path,
        default=script_dir,
        help=f"Test parameters (ops info) JSON file directory (default is {script_dir})",
    )
    parser.add_argument(
        "--convert-all-tests",
        action="store_true",
        help="Converts all tests instead of those picked by test_select",
    )
    parser.add_argument(
        "--keep-large-files",
        action="store_true",
        help="Keeps tests that contain files larger than 30MB in output directory",
    )
    parser.add_argument(
        "--capture-output",
        action="store_true",
        help="Prints output of running sh commands",
    )
    parser.add_argument(
        "-j",
        dest="num_cores",
        type=int,
        default=6,
        help="Number of simultaneous jobs to split the tasks into for multiprocessing",
    )
    parser.add_argument(
        "-v",
        dest="verbosity",
        action="count",
        default=0,
        help="Verbosity (can be used multiple times for more details)",
    )
    args = parser.parse_args(argv)

    return args


def main():
    args = parse_args()

    if not args.ref_model_dir.is_dir():
        logger.error(
            f"Missing or invalid reference model directory: {args.ref_model_dir}"
        )
        return 2
    else:
        ref_model = args.ref_model_dir / LOCATION_REF_MODEL_BINARY
        if not ref_model.is_file():
            logger.error(
                f"{LOCATION_REF_MODEL_BINARY} not found in {args.ref_model_dir}\nHave you built the reference model?"
            )
            return 2
    if args.unit_tests in ["framework", "both"]:
        if not args.framework_schema:
            logger.error(
                "Need to supply location of Framework flatbuffers schema via --framework-schema"
            )
            return 2
        if not args.framework_tests_dir.is_dir():
            logger.error(
                f"Missing or invalid framework tests directory: {args.framework_tests_dir}"
            )
            return 2

    loglevels = (logging.WARNING, logging.INFO, logging.DEBUG)
    loglevel = loglevels[min(args.verbosity, len(loglevels) - 1)]
    logger.setLevel(loglevel)
    # Set other loggers the same
    logging.getLogger("test_select").setLevel(loglevel)
    logging.getLogger("convert2conformance").setLevel(loglevel)

    print(f"Creating conformance tests for TOSA {args.profile} profile")
    print(f"Output directory: {args.output_dir}")

    if args.random_seed != DEFAULT_SEED:
        logger.warning(
            "Random test seed changed from default, tests will not match official conformance"
        )

    args.build_dir = args.build_dir.resolve()
    logger.debug(f"Creating build directory: {args.build_dir}")
    args.build_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Framework unit tests
        if args.unit_tests in ["framework", "both"]:
            logger.debug("Creating FRAMEWORK unit tests")
            test_picks_file = (
                args.param_json_dir / PROFILE_OPS_INFO[args.profile]["framework_tests"]
            )
            try:
                with open(test_picks_file, "r") as fd:
                    test_picks = json.load(fd)
            except Exception as e:
                logger.error(
                    f"Couldn't load framework tests info - {test_picks_file}: {e}"
                )
                return 1

            operators = args.operators
            if not operators:
                # Create tests for all the operators
                operators = list(test_picks.keys())

            root_output_dir = args.output_dir / "frameworks" / "tflite" / "operators"
            for op in operators:
                if op not in test_picks:
                    logger.warning(
                        f"Framework op {op} not found in {test_picks_file} - skipping"
                    )
                    continue

                logger.debug(f"Copying and renaming {op}")
                framework_test_dir = copy_rename_framework_tests(args, op, test_picks)
                profiles = test_picks[op]["profile"]
                if args.convert_all_tests:
                    logger.debug("Running and converting all framework tests")
                    convert_tests(
                        args,
                        op,
                        framework_test_dir,
                        root_output_dir,
                        profiles,
                        trim_op_subdir=True,
                    )
                else:
                    framework_tests = get_framework_tests_selection(
                        args, op, test_picks, framework_test_dir
                    )
                    convert_tests(
                        args,
                        op,
                        framework_test_dir,
                        root_output_dir,
                        profiles,
                        tests=framework_tests,
                        trim_op_subdir=True,
                    )

        # Operator unit tests
        if args.unit_tests in ["operator", "both"]:
            logger.debug("Creating OPERATOR unit tests")
            test_params_file = (
                args.param_json_dir
                / PROFILE_OPS_INFO[args.profile]["operator_test_params"]
            )
            try:
                with open(test_params_file, "r") as fd:
                    test_params = json.load(fd)
            except Exception as e:
                logger.error(
                    f"Couldn't load operator test params - {test_params_file}: {e}"
                )
                return 1

            operators = args.operators
            if not operators:
                # Create tests for all the operators
                operators = list(test_params.keys())

            for op in operators:
                if op not in test_params:
                    logger.warning(
                        f"{op} operator parameters not found in {test_params_file} - skipping"
                    )
                    continue

                if (
                    args.test_type == "negative"
                    and "no_negative_tests" in test_params[op]
                    and test_params[op]["no_negative_tests"]
                ):
                    logger.warning(f"No negative tests for {op}")
                    continue

                op_build_dir = build_op_tests(args, op, test_params)

                operator_group = test_params[op]["group"]
                root_output_dir = args.output_dir / "operators"
                profiles = test_params[op]["profile"]
                if args.convert_all_tests:
                    logger.debug(f"Running and converting all {op} tests")
                    generate_results(args, op, op_build_dir)
                    output_dir = convert_tests(
                        args,
                        op,
                        op_build_dir,
                        root_output_dir,
                        profiles,
                        group=operator_group,
                    )
                else:
                    if args.test_type in ["positive", "both"]:
                        tests_gen1, tests_gen2 = tee(
                            get_op_tests_selection(args, op, op_build_dir, test_params)
                        )
                        generate_results(args, op, op_build_dir, tests_gen1)
                        output_dir = convert_tests(
                            args,
                            op,
                            op_build_dir,
                            root_output_dir,
                            profiles,
                            tests=tests_gen2,
                            group=operator_group,
                        )
                    if args.test_type in ["negative", "both"] and (
                        "no_negative_tests" not in test_params[op]
                        or not test_params[op]["no_negative_tests"]
                    ):
                        negative_tests = get_op_tests_selection(
                            args, op, op_build_dir, test_params, negative=True
                        )
                        output_dir = convert_tests(
                            args,
                            op,
                            op_build_dir,
                            root_output_dir,
                            profiles,
                            tests=negative_tests,
                            group=operator_group,
                        )
                if not args.keep_large_files:
                    check_op_tests(args, op, output_dir)
    except GenConformanceError:
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

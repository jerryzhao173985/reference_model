#!/usr/bin/env python3
# Copyright (c) 2021-2025, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
"""Build conformance tests.

Steps:
- Specific input shapes (or tests) are specified and produced by using the
  settings in the .json files.
- Tests are selected to produce a good coverage.
- Tests are run on the reference model to produce the correct output files.
- Tests are converted to JSON and/or copied and saved to desired output directory.
"""
import argparse
import json
import logging
import multiprocessing as mp
import os
import re
import shlex
import shutil
import subprocess
from functools import partial
from pathlib import Path

import conformance.model_files as cmf
from conformance.tosa_profiles import TosaProfiles
from convert2conformance.convert2conformance import main as c2c_main
from convert2conformance.convert2conformance import OUTPUT_TYPE_DEFAULT
from convert2conformance.convert2conformance import OUTPUT_TYPES
from serializer.tosa_serializer import TOSA_VERSION

logging.basicConfig()
logger = logging.getLogger("tosa_verif_conformance_generator")

# Configuration
PROFILE_OPS_INFO = {
    "operator_test_params": "tosa_ext_profile_ops_info.json",
}

DEFAULT_SEED = 42

# When there is a dictionary of generator argument lists (groups) only the
# standard group will have negative tests generated for it
STANDARD_GENERATOR_GROUP = "standard"

TEST_VERSION_LATEST = "latest"
TEST_VERSION_V0_60_0 = "v0.60.0"
TEST_VERSIONS = (TEST_VERSION_LATEST, TEST_VERSION_V0_60_0)
REGEX_VERSION = re.compile(r"v([0-9]+)\.([0-9]+)\.([0-9]+)")


class GenConformanceError(Exception):
    """Generation error reporting exception."""

    pass


def _run_sh_command(args, cwd, full_cmd):
    """Run an external command and capture stdout/stderr."""
    # Quote the command line for printing
    try:
        full_cmd_esc = [shlex.quote(x) for x in full_cmd]
    except Exception as e:
        raise Exception(f"Error quoting command: {e}")
    if args.capture_output:
        logger.info(f"Command: {full_cmd_esc}")

    rc = subprocess.run(
        full_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd
    )

    if args.capture_output:
        stderr = rc.stderr.decode("utf-8")
        stdout = rc.stdout.decode("utf-8")
        logger.info(f"stderr: \n{stderr}")
        logger.info(f"stdout: \n{stdout}")
    if rc.returncode != 0:
        raise Exception(
            "Error running command: {}.\n{}".format(
                " ".join(full_cmd_esc), rc.stderr.decode("utf-8")
            )
        )
    return (rc.stdout, rc.stderr)


def build_op_tests(
    args,
    test_type,
    profile_ext,
    operator,
    group,
    gen_args_list,
    gen_neg_dim_range,
    supports=[],
    gen_filter=None,
    selector_info=None,
):
    """Build tests for a given operator.

    Builds a set of tests based on the given generator arguments list

    Returns operator output directory
    """
    build_tests_cmd = "tosa_verif_build_tests"
    op_build_dir = args.build_dir / profile_ext / group

    if gen_filter is None:
        gen_filter = f"^{operator}$"

    build_cmd_base = [
        build_tests_cmd,
        "--generate-lib-path",
        str(args.generate_lib_path),
        "--filter",
        gen_filter,
        "-o",
        str(op_build_dir),
        "--seed",
        str(args.random_seed),
    ]
    if args.verbosity:
        build_cmd_base.append("-" + ("v" * args.verbosity))

    if args.tests_list_file is not None:
        build_cmd_base.append("--list-tests")

    if "lazy_data_gen" in supports and args.lazy_data_generation:
        build_cmd_base.append("--lazy-data-generation")

    if "stable_random_gen" in supports and not args.global_random_generation:
        build_cmd_base.append("--stable-random-generation")

    if "random_const_inputs" in supports:
        build_cmd_base.append("--random-const-inputs")

    # Always use the new generator_select mode for conformance
    if selector_info is None:
        logger.error(
            "build_op_tests error: generator_select mode without selector information"
        )
        raise (GenConformanceError())
    selector_config, selector_name = selector_info
    build_cmd_base.extend(
        [
            "--test-selection-config",
            str(selector_config),
            "--test-selection-criteria",
            selector_name,
        ]
    )

    # Add extra profile/extension info to allow test filtering
    build_cmd_base.append("--profile")
    build_cmd_base.extend(args.profile)
    build_cmd_base.append("--extension")
    if len(args.extension) == 0:
        build_cmd_base.append(TosaProfiles.PROFILES_EXTENSIONS_NONE)
    else:
        build_cmd_base.extend(args.extension)

    build_cmds_list = []

    if test_type in ["positive", "both"]:
        # Append extra parameters and run test generator for each set of parameters.
        for arglist in gen_args_list:
            build_cmd_pos_test = build_cmd_base.copy()
            build_cmd_pos_test.extend(["--test-type", "positive"])
            build_cmd_pos_test.extend(arglist)
            build_cmds_list.append(build_cmd_pos_test)

    if test_type in ["negative", "both"]:
        # Get target-dtypes options and any filter string to limit tests
        target_dtypes_args = []
        for arglist in gen_args_list:
            idx = 0
            while idx < len(arglist):
                if arglist[idx] == "--target-dtype":
                    idx += 1
                    # Support single or multiple args after --target-dtype
                    while idx < len(arglist) and (not arglist[idx].startswith("--")):
                        if arglist[idx] not in target_dtypes_args:
                            target_dtypes_args.append(arglist[idx])
                        idx += 1
                else:
                    idx += 1
        build_cmd_neg_test = build_cmd_base.copy()
        build_cmd_neg_test.extend(["--test-type", "negative"])
        # Limit sizes of negative tests
        dim_range = gen_neg_dim_range if gen_neg_dim_range is not None else "1,16"
        build_cmd_neg_test.extend(["--tensor-dim-range", dim_range])
        for arg in target_dtypes_args:
            build_cmd_neg_test.extend(["--target-dtype", arg])
        build_cmds_list.append(build_cmd_neg_test)

    logger.info(f"Processing {operator} tests in {len(build_cmds_list)} batch(es)")
    error = False
    for i, cmd in enumerate(build_cmds_list):
        try:
            raw_stdout, _ = _run_sh_command(args, args.ref_model_path.parent, cmd)
            logger.info(
                f"{operator} test batch {(i + 1)}/{len(build_cmds_list)} completed successfully"
            )

            if args.tests_list_file is not None:
                with args.tests_list_file.open("a") as fd:
                    fd.write(raw_stdout.decode("utf-8"))

        except Exception as e:
            logger.error(
                f"{operator} test batch {(i + 1)}/{len(build_cmds_list)} unsuccessful, skipping"
            )
            logger.error(f" build_op_tests error: {e} ")
            error = True
    if error:
        raise (GenConformanceError())

    return op_build_dir


def _check_to_include_test(test_type, test_name):
    """Check test name for inclusion based on test_type, returns True to include."""

    if test_type == "both":
        return True
    else:
        error_test = "_ERRORIF_" in test_name
        return (error_test and test_type == "negative") or (
            not error_test and test_type == "positive"
        )


def _get_all_tests_list(test_type, test_root_dir, operator):
    """Create test list from tests in the test_dir based on chosen type."""
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
                if _check_to_include_test(test_type, test.name)
            ]
        )
    return tests


def generate_results(
    args, profile_ext, operator, op_build_dir, supports=[], tests=None
):
    """Run tests on reference model and save result to the test directory."""
    if "lazy_data_gen" in supports and args.lazy_data_generation:
        logger.info("Skipping running tests due to lazy data gen")
        return

    num_cores = args.num_cores

    # Use the test runner
    ref_cmd_base = [
        "tosa_verif_run_tests",
        "--ref-model-path",
        str(args.ref_model_path),
        "--schema-path",
        str(args.schema_path),
        "-j",
        str(num_cores),
        "-v",
        "-t",
    ]
    ref_cmds = []

    if not tests:
        # Do not need to run ERRORIF tests as they don't have result files
        tests = _get_all_tests_list("positive", op_build_dir, operator)

    skipped = 0
    for test in tests:
        desc = test / "desc.json"
        with desc.open("r") as fd:
            test_desc = json.load(fd)
        if "meta" in test_desc and "compliance" in test_desc["meta"]:
            skipped += 1
            logger.debug(
                f"Skipping generating results for new compliance test - {str(test)}"
            )
            continue
        ref_cmd = ref_cmd_base.copy()
        ref_cmd.append(str(test.absolute()))
        ref_cmds.append(ref_cmd)

    if skipped:
        logger.info(f"{skipped} new compliance tests skipped for results generation")

    fail_string = "UNEXPECTED_FAILURE"
    failed_counter = 0

    job_pool = mp.Pool(args.num_cores)
    sh_partial = partial(_run_sh_command, args, args.ref_model_path.parent)
    pool_results = job_pool.map(sh_partial, ref_cmds)
    job_pool.close()
    job_pool.join()

    # Use captured output for run_sh_command to work out if test passed.
    for i, rc in enumerate(pool_results):
        if fail_string in str(rc[0]):
            logger.error(f"Test {i + 1}/{len(ref_cmds)}: {ref_cmds[i][-1]} failed.")
            failed_counter += 1
        else:
            logger.debug(f"Test {i + 1}/{len(ref_cmds)}: {ref_cmds[i][-1]} passed.")

    logger.info(f"{len(ref_cmds) - failed_counter}/{len(ref_cmds)} tests passed")
    logger.info("Ran tests on model and saved results of passing tests")


def convert_tests(
    args,
    test_type,
    profile_ext,
    operator,
    op_build_dir,
    output_dir,
    op_profiles_extensions_list,
    supports=[],
    tests=None,
    group=None,
    trim_op_subdir=False,
    tags=None,
):
    """Convert/copy tests to output directory."""
    if group:
        output_dir = output_dir / group

    c2c_args_base = ["--strict"]
    c2c_args_base.extend(["--schema-path", str(args.schema_path)])
    c2c_args_base.extend(["--flatc-path", str(args.flatc_path)])
    c2c_args_base.extend(["--output-type", args.output_type])

    if tags is not None:
        for tag in tags:
            c2c_args_base.extend(["--tag", tag])
    if "lazy_data_gen" in supports and args.lazy_data_generation:
        lazy_data_gen = True
        c2c_args_base.append("--lazy-data-generation")
    c2c_args_base.append("--output-directory")

    c2c_args_list = []

    if not tests:
        tests = _get_all_tests_list(test_type, op_build_dir, operator)
        logger.info(f"Converting all {profile_ext} profile tests of type {test_type}")

    # Controls if we copy the tests in their operator sub-directory or not
    output_dir_relative_pos = -1 if trim_op_subdir else -2
    for test in tests:
        logger.debug(f"Test chosen: {test}")
        c2c_args = c2c_args_base.copy()
        full_output_directory = output_dir / test.relative_to(
            *test.parts[:output_dir_relative_pos]
        )
        c2c_args.append(str(full_output_directory))
        c2c_args.append(str(test))
        c2c_args_list.append(c2c_args)

    if len(c2c_args_list) == 0:
        if lazy_data_gen:
            # TODO - remove this when all lazy_gen_tests can be produced
            logger.warning(
                f"Tests missing for {operator} in {op_build_dir}. See verbose output for more info."
            )
        else:
            logger.error(
                f"No tests found for {operator}. Nothing to convert in {op_build_dir}"
            )
            raise (GenConformanceError())

    job_pool = mp.Pool(args.num_cores)

    pool_results = job_pool.map(c2c_main, c2c_args_list)
    job_pool.close()
    job_pool.join()

    failed_counter = 0
    for i, result in enumerate(pool_results):
        if result != 0:
            logger.error(
                f"test {i + 1}/{len(c2c_args_list)}: {c2c_args_list[i][-1]} failed to convert."
            )
            failed_counter += 1
        else:
            logger.debug(
                f"test {i + 1}/{len(c2c_args_list)}: {c2c_args_list[i][-1]} converted"
            )
    logger.info(
        f"{len(c2c_args_list) - failed_counter}/{len(c2c_args_list)} tests successfully converted"
    )

    if failed_counter > 0:
        logger.error(f"Stopping due to {failed_counter} test conversion errors")
        raise (GenConformanceError())

    logger.info("Converted/copied tests and saved to output directory")

    return output_dir


def check_op_tests_size(args, profile, operator, output_dir):
    """Move test folders than contain files larger than a specified size to new directory."""
    limit_in_mb = args.large_file_limit
    if limit_in_mb <= 0:
        # Nothing to do - no limit set
        return

    destination_dir = str(args.output_dir) + "_large_files"

    # Include all tests - both positive and negative
    tests = _get_all_tests_list("both", output_dir, operator)
    if not tests:
        logger.error(
            f"Couldn't find any tests to size check for {operator} in {output_dir}"
        )
        raise (GenConformanceError())

    for tdir in tests:
        move_dir = False
        test_files = [file for file in tdir.glob("*")]
        for file in test_files:
            file_size_in_mb = os.stat(file).st_size / 1024**2
            if file_size_in_mb > limit_in_mb:
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


def parse_args(argv=None):
    """Parse the arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--operators",
        "--op",
        type=str,
        nargs="*",
        help="The operator(s) to create tests for, if not supplied all tests will be created",
    )
    # Add --profile and --extension options
    TosaProfiles.addArgumentsToParser(parser, all_extensions_default=False)

    parser.add_argument(
        "--unit-tests",
        dest="unit_tests",
        choices=["operator"],
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
        "--global-random-generation",
        action="store_true",
        help="Disable stable random generation of tests that support this mode",
    )
    parser.add_argument(
        "--lazy-data-generation",
        action="store_true",
        help=f"Enable lazy data generation (only for {TosaProfiles.TosaProFP})",
    )
    rm_group = parser.add_mutually_exclusive_group(required=True)
    rm_group.add_argument(
        "--ref-model-directory",
        dest="ref_model_dir",
        type=Path,
        help="(DEPRECATED - use ref-model-path) Reference Model directory - with build directory",
    )
    rm_group.add_argument(
        "--ref-model-path",
        dest="ref_model_path",
        type=Path,
        help="Path to TOSA reference model executable",
    )
    parser.add_argument(
        "--generate-lib-path",
        dest="generate_lib_path",
        type=Path,
        help=(
            "Path to TOSA generate library. Defaults to "
            "the library in the directory of `ref-model-path`"
        ),
    )
    parser.add_argument(
        "--schema-path",
        "--operator-fbs",
        dest="schema_path",
        type=Path,
        help=(
            "Path to TOSA reference model flat buffer schema. Defaults to "
            f"`{cmf.DEFAULT_REF_MODEL_SCHEMA_PATH}` in parents parent directory of `ref-model-path`"
        ),
    )
    parser.add_argument(
        "--flatc-path",
        dest="flatc_path",
        type=Path,
        help=(
            "Path to flatc executable. Defaults to "
            f"`{cmf.DEFAULT_REF_MODEL_BUILD_FLATC_PATH}` in parent directory of `ref-model-path`"
        ),
    )
    parser.add_argument(
        "--test-version",
        dest="test_version",
        choices=TEST_VERSIONS,
        default=TEST_VERSION_LATEST,
        help=f"Version of the tests to produce (default is {TEST_VERSION_LATEST})",
    )
    parser.add_argument(
        "--output-type",
        dest="output_type",
        choices=OUTPUT_TYPES,
        default=OUTPUT_TYPE_DEFAULT,
        help=f"Output file type produced (default is {OUTPUT_TYPE_DEFAULT})",
    )
    parser.add_argument(
        "--seed",
        dest="random_seed",
        default=DEFAULT_SEED,
        type=int,
        help="Random test seed",
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
        "--test-params-json-config",
        "--config",
        dest="param_config",
        type=Path,
        help="Test parameters (ops info) JSON file (overrides --test-param-json-directory)",
    )
    parser.add_argument(
        "--convert-all-tests",
        action="store_true",
        help="Converts all tests instead of those picked by test_select",
    )
    parser.add_argument(
        "--list-tests-to-file",
        dest="tests_list_file",
        type=Path,
        help="Lists out the tests to be generated to a file instead of generating them",
    )
    parser.add_argument(
        "--keep-large-files",
        action="store_true",
        help="[DEPRECATED] All files now kept by default unless --large-file-limit is set",
    )
    parser.add_argument(
        "--large-file-limit",
        type=int,
        default=0,
        help="Size in megabytes that limits conformance data files, tests exceeding this will be moved from output",
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

    if args.ref_model_dir is not None:
        # Assume the ref model exe path based on the ref model directory
        args.ref_model_path = cmf.find_tosa_file(
            cmf.TosaFileType.REF_MODEL, args.ref_model_dir, False
        )
    if not args.ref_model_path.is_file():
        logger.error(
            f"Missing reference model binary (--ref-model-path): {args.ref_model_path}"
        )
        return None
    args.ref_model_path = args.ref_model_path.absolute()

    if args.generate_lib_path is None:
        args.generate_lib_path = cmf.find_tosa_file(
            cmf.TosaFileType.GENERATE_LIBRARY, args.ref_model_path
        )
    if not args.generate_lib_path.is_file():
        logger.error(
            f"Missing TOSA generate data library (--generate-lib-path): {args.generate_lib_path}"
        )
        return None
    args.generate_lib_path = args.generate_lib_path.absolute()

    if args.schema_path is None:
        args.schema_path = cmf.find_tosa_file(
            cmf.TosaFileType.SCHEMA, args.ref_model_path
        )
    if not args.schema_path.is_file():
        logger.error(
            f"Missing reference model schema (--schema-path): {args.schema_path}"
        )
        return None
    args.schema_path = args.schema_path.absolute()

    if args.flatc_path is None:
        args.flatc_path = cmf.find_tosa_file(
            cmf.TosaFileType.FLATC, args.ref_model_path
        )
    if not args.flatc_path.is_file():
        logger.error(f"Missing flatc binary (--flatc-path): {args.flatc_path}")
        return None
    args.flatc_path = args.flatc_path.absolute()

    args.param_json_dir = args.param_json_dir.absolute()

    if args.param_config is not None:
        args.param_config = args.param_config.absolute()

    # Update/validate the --profile and --extension options
    TosaProfiles.parseArguments(args, logger)

    return args


def in_version(test_version, gen_dict):
    """Check if the selected test_version is compatible with the tests."""

    def version_string_to_numbers(verstr):
        # Turn the "vM.mm.pp" string into Major, Minor, Patch versions
        if verstr == TEST_VERSION_LATEST:
            return (TOSA_VERSION[0], TOSA_VERSION[1], TOSA_VERSION[2])
        else:
            match = re.match(REGEX_VERSION, verstr)
            if match is None:
                raise KeyError(f"Invalid version string {verstr}")
            return (int(v) for v in match.groups())

    if "from_version" in gen_dict:
        selected_version = version_string_to_numbers(test_version)
        from_version = version_string_to_numbers(gen_dict["from_version"])

        # Check the Major version is compatible, then Minor, and lastly Patch
        # Unless the versions match, we can exit early due to obvious precedence
        for sel, fro in zip(selected_version, from_version):
            if sel < fro:
                # From version is later than selected version
                return False
            elif sel > fro:
                # From version is earlier than selected version
                return True
        # If we get here, the version numbers match exactly
        return True
    else:
        # No specific version info
        return True


def _get_log_level(verbosity):
    loglevels = (logging.WARNING, logging.INFO, logging.DEBUG)
    verbosity = max(verbosity, 0)
    return loglevels[min(verbosity, len(loglevels) - 1)]


def main():
    args = parse_args()
    if args is None:
        # Argument processing error
        return 2

    loglevel = _get_log_level(args.verbosity)
    logger.setLevel(loglevel)
    # Set other loggers to a quieter level
    loglevel = _get_log_level(args.verbosity - 1)
    logging.getLogger("test_select").setLevel(loglevel)
    logging.getLogger("convert2conformance").setLevel(loglevel)

    if args.random_seed != DEFAULT_SEED:
        logger.warning(
            "Random test seed changed from default, tests will not match official conformance"
        )

    if args.tests_list_file is not None:
        # Try creating tests list file
        with args.tests_list_file.open("w") as fd:
            fd.write("")
        action = "Listing"
    else:
        print(f"Output directory: {args.output_dir}")
        args.build_dir = args.build_dir.resolve()
        logger.debug(f"Creating build directory: {args.build_dir}")
        args.build_dir.mkdir(parents=True, exist_ok=True)
        action = "Creating"

    profileExtList = args.profile + args.extension
    profileExtDone = []

    try:
        for profile_ext in profileExtList:
            # Operator unit tests
            if args.unit_tests in ("operator",):
                logger.debug(f"{action} OPERATOR unit tests")
                if args.param_config is None:
                    # Use default config
                    config = PROFILE_OPS_INFO["operator_test_params"]
                    test_params_file = args.param_json_dir / config
                else:
                    test_params_file = args.param_config

                try:
                    with open(test_params_file, "r") as fd:
                        test_params = json.load(fd)
                except Exception as e:
                    logger.error(
                        f"Couldn't load operator test params - {test_params_file}: {e}"
                    )
                    return 1
                logger.debug(f"Using config file: {str(test_params_file)}")

                operators = args.operators
                if not operators:
                    # Create tests for all the operators
                    operators = list(test_params.keys())

                print(
                    f"{action} conformance tests for TOSA {profile_ext} profile/extension"
                )

                # Use a set to ignore duplicate operators chosen
                for op in set(operators):
                    logger.info(f"OPERATOR: {op}")
                    if op not in test_params:
                        logger.warning(
                            f"{op} operator parameters not found in {test_params_file} - skipping"
                        )
                        continue

                    operator_group = test_params[op]["group"]
                    root_output_dir = args.output_dir / "operators"
                    supports = test_params[op].get("support_for", [])
                    gen_filter = test_params[op].get("gen_filter", None)
                    old_profile_info = test_params[op].get("profile", [])

                    # Iterate through the generation groups selecting tests from each
                    for gen_name, gen_dict in test_params[op]["generation"].items():
                        supports_any = gen_dict.get("supports_any", [])
                        supports_all = gen_dict.get("supports_all", [])

                        # Fall back for old configs
                        if not supports_all and not supports_any:
                            if not old_profile_info:
                                logger.error(
                                    f"generator {gen_name} for {op} is missing supports_all/supports_any"
                                )
                                raise (GenConformanceError())
                            else:
                                supports_any = old_profile_info

                        supported = supports_any + supports_all

                        if profile_ext not in supported:
                            logger.info(
                                f"No match for profile/extension {profile_ext} for generation group {gen_name} - skipping"
                            )
                            continue

                        if any(p in supported for p in profileExtDone):
                            logger.info(
                                f"Already used this generator {gen_name} before - skipping"
                            )
                            continue

                        # Already checked that the profile is in supported (any or all), so this
                        # verifies that all conditions are met
                        if not all(p in profileExtList for p in supports_all):
                            logger.info(
                                f"For profile/extension {profile_ext} the profiles/extensions chosen do not meet all the requirements of {supports_all} - skipping"
                            )
                            continue

                        if not in_version(args.test_version, gen_dict):
                            logger.warning(
                                f"{op} [{gen_name}] is not in {args.test_version} - skipping"
                            )
                            continue

                        no_neg_tests = (
                            "no_negative_tests" in gen_dict
                            and gen_dict["no_negative_tests"] == "true"
                        )

                        if no_neg_tests:
                            if args.test_type == "negative":
                                logger.info(
                                    f"No negative tests for {op} / generation group {gen_name}"
                                )
                                continue
                            # Only produce positive tests
                            test_type = "positive"
                        else:
                            test_type = args.test_type

                        gen_neg_dim_range = (
                            gen_dict["negative_dim_range"]
                            if "negative_dim_range" in gen_dict
                            else None
                        )

                        # Work out which selection criteria we are using
                        if "selector" in gen_dict:
                            selector_name = gen_dict["selector"]
                            if selector_name not in test_params[op]["selection"]:
                                logger.warn(
                                    f"Could not find {selector_name} in selection dict for {op} - using default"
                                )
                                selector_name = "default"
                        else:
                            selector_name = "default"

                        if selector_name not in test_params[op]["selection"]:
                            logger.error(
                                f"Could not find {selector_name} in selection dict for {op}"
                            )
                            raise (GenConformanceError())

                        op_build_dir = build_op_tests(
                            args,
                            test_type,
                            profile_ext,
                            op,
                            gen_name,
                            gen_dict["generator_args"],
                            gen_neg_dim_range,
                            supports=supports,
                            gen_filter=gen_filter,
                            selector_info=(test_params_file, selector_name),
                        )

                        if args.tests_list_file is not None:
                            logger.info("Tests list file extended")
                            continue

                        if test_type in ["positive", "both"]:
                            logger.info(f"Running and converting all {op} tests")
                            generate_results(
                                args,
                                profile_ext,
                                op,
                                op_build_dir,
                                supports=supports,
                            )
                        operator_test_list = None

                        tags = (
                            [gen_name] if gen_name != STANDARD_GENERATOR_GROUP else None
                        )
                        output_dir = convert_tests(
                            args,
                            test_type,
                            profile_ext,
                            op,
                            op_build_dir,
                            root_output_dir,
                            supported,
                            supports=supports,
                            tests=operator_test_list,
                            group=operator_group,
                            tags=tags,
                        )

                        check_op_tests_size(args, profile_ext, op, output_dir)

            profileExtDone.append(profile_ext)

    except GenConformanceError:
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

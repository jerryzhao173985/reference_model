# Copyright (c) 2020-2025, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import logging
import re
import sys
from pathlib import Path

import conformance.model_files as cmf
import generator.tosa_test_select as tts
from conformance.tosa_profiles import TosaProfiles
from generator.tosa_test_gen import TosaTestGen
from serializer.tosa_serializer import dtype_str_to_val
from serializer.tosa_serializer import DTypeNames

OPTION_FP_VALUES_RANGE = "--fp-values-range"
PROFILES_EXTENSIONS_ALL = "all"
PROFILES_EXTENSIONS_NONE = "none"

logging.basicConfig()
logger = logging.getLogger("tosa_verif_build_tests")


# Used for parsing a comma-separated list of integers/floats in a string
# to an actual list of integers/floats with special case max
def str_to_list(in_s, is_float=False):
    """Converts a comma-separated list string to a python list of numbers."""
    lst = in_s.split(",")
    out_list = []
    for i in lst:
        # Special case for allowing maximum FP numbers
        if is_float and i in ("-max", "max"):
            val = i
        else:
            val = float(i) if is_float else int(i)
        out_list.append(val)
    return out_list


def auto_int(x):
    """Converts hex/dec argument values to an int"""
    return int(x, 0)


def parseArgs(argv):
    """Parse the command line arguments."""
    if argv is None:
        argv = sys.argv[1:]

    if OPTION_FP_VALUES_RANGE in argv:
        # Argparse fix for hyphen (minus values) in argument values
        # convert "ARG VAL" into "ARG=VAL"
        # Example --fp-values-range -2.0,2.0 -> --fp-values-range=-2.0,2.0
        new_argv = []
        idx = 0
        while idx < len(argv):
            arg = argv[idx]
            if arg == OPTION_FP_VALUES_RANGE and idx + 1 < len(argv):
                val = argv[idx + 1]
                if val.startswith("-"):
                    arg = f"{arg}={val}"
                    idx += 1
            new_argv.append(arg)
            idx += 1
        argv = new_argv

    parser = argparse.ArgumentParser()

    filter_group = parser.add_argument_group("test filter options")
    ops_group = parser.add_argument_group("operator options")
    tens_group = parser.add_argument_group("tensor options")

    parser.add_argument(
        "-o", dest="output_dir", type=str, default="vtest", help="Test output directory"
    )

    parser.add_argument(
        "--seed",
        dest="random_seed",
        default=42,
        type=int,
        help="Random seed for test generation",
    )

    parser.add_argument(
        "--stable-random-generation",
        dest="stable_rng",
        action="store_true",
        help="Produces less variation (when the test-generator changes) in the test output using the same options",
    )

    filter_group.add_argument(
        "--filter",
        dest="filter",
        default="",
        type=str,
        help="Filter operators by this regular expression (all operator names are lower case)",
    )

    filter_group.add_argument(
        "--profile",
        dest="profile",
        choices=TosaProfiles.profiles() + [PROFILES_EXTENSIONS_ALL],
        default=[PROFILES_EXTENSIONS_ALL],
        type=str,
        nargs="*",
        help=f"TOSA profile(s) - used for filtering CAST operations (default is {PROFILES_EXTENSIONS_ALL})",
    )

    filter_group.add_argument(
        "--extension",
        dest="extension",
        choices=TosaProfiles.extensions()
        + [PROFILES_EXTENSIONS_ALL, PROFILES_EXTENSIONS_NONE],
        default=[PROFILES_EXTENSIONS_ALL],
        type=str,
        nargs="*",
        help=f"TOSA extension(s) - used for filtering CAST operations (default is {PROFILES_EXTENSIONS_ALL})."
        + f" Use {PROFILES_EXTENSIONS_NONE} to choose no extensions.",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="count",
        default=0,
        help="Verbose operation",
    )

    parser.add_argument(
        "--lazy-data-generation",
        dest="lazy_data_gen",
        action="store_true",
        help="Tensor data generation is delayed til test running",
    )

    parser.add_argument(
        "--generate-lib-path",
        dest="generate_lib_path",
        type=Path,
        help="Path to TOSA generate library.",
    )

    # Constraints on tests
    tens_group.add_argument(
        "--tensor-dim-range",
        dest="tensor_shape_range",
        default="1,64",
        type=lambda x: str_to_list(x),
        help="Min,Max range of tensor shapes",
    )

    tens_group.add_argument(
        OPTION_FP_VALUES_RANGE,
        dest="tensor_fp_value_range",
        default="0.0,1.0",
        type=lambda x: str_to_list(x, is_float=True),
        help="Min,Max range of floating point tensor values",
    )

    ops_group.add_argument(
        "--max-batch-size",
        dest="max_batch_size",
        default=1,
        type=positive_integer_type,
        help="Maximum batch size for NHWC tests",
    )

    ops_group.add_argument(
        "--max-conv-padding",
        dest="max_conv_padding",
        default=1,
        type=int,
        help="Maximum padding for Conv tests",
    )

    ops_group.add_argument(
        "--max-conv-dilation",
        dest="max_conv_dilation",
        default=2,
        type=int,
        help="Maximum dilation for Conv tests",
    )

    ops_group.add_argument(
        "--max-conv-stride",
        dest="max_conv_stride",
        default=2,
        type=int,
        help="Maximum stride for Conv tests",
    )

    ops_group.add_argument(
        "--conv-kernel",
        dest="conv_kernels",
        action="extend",
        default=[],
        type=lambda x: str_to_list(x),
        nargs="*",
        help="Create convolution tests with a particular kernel shape, e.g., 1,4 or 1,3,1 (only 2D kernel sizes will be used for 2D ops, etc.)",
    )

    ops_group.add_argument(
        "--max-pooling-padding",
        dest="max_pooling_padding",
        default=1,
        type=int,
        help="Maximum padding for pooling tests",
    )

    ops_group.add_argument(
        "--max-pooling-stride",
        dest="max_pooling_stride",
        default=2,
        type=int,
        help="Maximum stride for pooling tests",
    )

    ops_group.add_argument(
        "--max-pooling-kernel",
        dest="max_pooling_kernel",
        default=3,
        type=int,
        help="Maximum kernel for pooling tests",
    )

    ops_group.add_argument(
        "--num-rand-permutations",
        dest="num_rand_permutations",
        default=6,
        type=int,
        help="Number of random permutations for a given shape/rank for randomly-sampled parameter spaces",
    )

    ops_group.add_argument(
        "--max-resize-output-dim",
        dest="max_resize_output_dim",
        default=1000,
        type=int,
        help="Upper limit on width and height output dimensions for `resize` op. Default: 1000",
    )

    # Targeting a specific shape/rank/dtype
    tens_group.add_argument(
        "--target-shape",
        dest="target_shapes",
        action="extend",
        default=[],
        # Used for parsing a comma-separated list of integers in a string
        type=lambda x: str_to_list(x),
        nargs="*",
        help="Create tests with a particular input tensor shape, e.g., 1,4,4,8 (may be repeated for tests that require multiple input shapes)",
    )

    tens_group.add_argument(
        "--target-rank",
        dest="target_ranks",
        action="extend",
        default=None,
        type=lambda x: auto_int(x),
        nargs="*",
        help="Create tests with a particular input tensor rank (may be repeated)",
    )

    tens_group.add_argument(
        "--target-dtype",
        dest="target_dtypes",
        action="extend",
        default=None,
        type=lambda x: dtype_str_to_val(x),
        nargs="*",
        help=f"Create test with a particular DType: [{', '.join([d.lower() for d in DTypeNames[1:]])}] (may be repeated)",
    )

    ops_group.add_argument(
        "--random-const-inputs",
        dest="random_const_inputs",
        action="store_true",
        help="Allow any combination of input/constant tensors for operators",
    )

    ops_group.add_argument(
        "--num-const-inputs-concat",
        dest="num_const_inputs_concat",
        default=0,
        choices=[0, 1, 2, 3],
        type=int,
        help="Allow constant input tensors for concat operator",
    )

    filter_group.add_argument(
        "--test-type",
        dest="test_type",
        choices=["positive", "negative", "both"],
        default="positive",
        type=str,
        help="type of tests produced, positive, negative, or both",
    )

    filter_group.add_argument(
        "--test-selection-config",
        dest="selection_config",
        type=Path,
        help="enables test selection, this is the path to the JSON test selection config file, will use the default selection specified for each op unless --selection-criteria is supplied",
    )

    filter_group.add_argument(
        "--test-selection-criteria",
        dest="selection_criteria",
        help="enables test selection, this is the selection criteria to use from the selection config",
    )

    filter_group.add_argument(
        "--no-special-tests",
        dest="no_special_tests",
        action="store_true",
        help="Do not produce special 'full range' or 'FP special' tests",
    )

    parser.add_argument(
        "--list-tests",
        dest="list_tests",
        action="store_true",
        help="lists the tests that will be generated and then exits",
    )

    ops_group.add_argument(
        "--oversize",
        "--allow-pooling-and-conv-oversizes",
        dest="oversize",
        action="store_true",
        help="allow oversize padding, stride and kernel tests",
    )

    ops_group.add_argument(
        "--zero-point",
        dest="zeropoint",
        default=None,
        type=int,
        help="set a particular zero point for all valid positive tests",
    )

    parser.add_argument(
        "--dump-const-tensors",
        dest="dump_consts",
        action="store_true",
        help="output const tensors as numpy files for inspection",
    )

    ops_group.add_argument(
        "--level-8k-sizes",
        dest="level8k",
        action="store_true",
        help="create level 8k size tests",
    )

    args = parser.parse_args(argv)

    if PROFILES_EXTENSIONS_ALL in args.profile:
        args.profile = TosaProfiles.profiles()

    if PROFILES_EXTENSIONS_ALL in args.extension:
        args.extension = TosaProfiles.extensions()
    elif PROFILES_EXTENSIONS_NONE in args.extension:
        args.extension = []

    return args


def positive_integer_type(argv_str):
    value = int(argv_str)
    if value <= 0:
        msg = f"{argv_str} is not a valid positive integer"
        raise argparse.ArgumentTypeError(msg)
    return value


def main(argv=None):

    args = parseArgs(argv)

    loglevels = (logging.WARNING, logging.INFO, logging.DEBUG)
    loglevel = loglevels[min(args.verbose, len(loglevels) - 1)]
    logger.setLevel(loglevel)

    if not args.lazy_data_gen:
        if args.generate_lib_path is None:
            args.generate_lib_path = cmf.find_tosa_file(
                cmf.TosaFileType.GENERATE_LIBRARY, Path("reference_model"), False
            )
        if not args.generate_lib_path.is_file():
            print(
                f"Argument error: Generate library (--generate-lib-path) not found - {str(args.generate_lib_path)}"
            )
            return 2

    ttg = TosaTestGen(args)

    # Determine if test selection mode is enabled or not
    selectionMode = (
        args.selection_config is not None or args.selection_criteria is not None
    )
    selectionCriteria = (
        "default" if args.selection_criteria is None else args.selection_criteria
    )
    if args.selection_config is not None:
        # Try loading the selection config
        if not args.generate_lib_path.is_file():
            print(
                f"Argument error: Test selection config (--test-selection-config) not found {str(args.selection_config)}"
            )
            return 2
        with args.selection_config.open("r") as fd:
            selectionCfg = json.load(fd)
    else:
        # Fallback to using anything defined in the TosaTestGen list
        selectionCfg = ttg.TOSA_OP_LIST
        # Set up some defaults to create a quick testing selection
        selectDefault = {"default": {"permutes": ["rank", "type"], "maximum": 10}}
        for opName in selectionCfg:
            if (
                "selection" not in selectionCfg[opName]
                or "default" not in selectionCfg[opName]["selection"]
            ):
                selectionCfg[opName]["selection"] = selectDefault

    if args.test_type == "both":
        testType = ["positive", "negative"]
    else:
        testType = [args.test_type]

    results = []
    for test_type in testType:
        testList = tts.TestList(selectionCfg, selectionCriteria=selectionCriteria)
        try:
            for operator in ttg.TOSA_OP_LIST:
                name = ttg.getOperatorNameStr(operator)
                if re.match(args.filter + ".*", name):
                    tests = ttg.genOpTestList(
                        operator,
                        shapeFilter=args.target_shapes,
                        rankFilter=args.target_ranks,
                        dtypeFilter=args.target_dtypes,
                        testType=test_type,
                    )
                    for testOp, testStr, dtype, error, shapeList, argsDict in tests:
                        testOpName = ttg.getOperatorNameStr(testOp)
                        test = tts.Test(
                            testOpName,
                            testStr,
                            dtype,
                            error,
                            shapeList,
                            argsDict,
                            testOp,
                        )
                        testList.add(test)
        except Exception as e:
            logger.error(
                f"INTERNAL ERROR: Failure generating test lists for {operator}"
            )
            raise e

        if not selectionMode:
            # Allow all tests to be selected
            tests = testList.all(
                profiles_chosen=args.profile,
                extensions_chosen=args.extension,
            )
        else:
            # Use the random number generator to shuffle the test list
            # and select the per op tests from it
            tests = testList.select(
                rng=ttg.global_rng,
                profiles_chosen=args.profile,
                extensions_chosen=args.extension,
            )

        if args.list_tests:
            for test in tests:
                print(test)
            continue

        print(f"{len(tests)} matching {test_type} tests")

        try:
            for test in tests:
                opName = test.testOpName
                results.append(
                    ttg.serializeTest(
                        opName,
                        str(test),
                        test.dtype,
                        test.error,
                        test.shapeList,
                        test.argsDict,
                    )
                )
        except Exception as e:
            logger.error(f"INTERNAL ERROR: Failure creating test output for {opName}")
            raise e

    if results.count(False):
        raise Exception(f"Failed to create {results.count(False)} tests")

    if not args.list_tests:
        print(f"Done creating {len(results)} tests")
    return 0


if __name__ == "__main__":
    exit(main())

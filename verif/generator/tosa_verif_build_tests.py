# Copyright (c) 2020-2022, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
import argparse
import re

from generator.tosa_test_gen import TosaTestGen
from serializer.tosa_serializer import dtype_str_to_val


# Used for parsing a comma-separated list of integers in a string
# to an actual list of integers
def str_to_list(in_s):
    """Converts a comma-separated list of string integers to a python list of ints"""
    lst = in_s.split(",")
    out_list = []
    for i in lst:
        out_list.append(int(i))
    return out_list


def auto_int(x):
    """Converts hex/dec argument values to an int"""
    return int(x, 0)


def parseArgs(argv):

    parser = argparse.ArgumentParser()
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
        "--filter",
        dest="filter",
        default="",
        type=str,
        help="Filter operator test names by this expression",
    )

    parser.add_argument(
        "-v", "--verbose", dest="verbose", action="count", help="Verbose operation"
    )

    # Constraints on tests
    parser.add_argument(
        "--tensor-dim-range",
        dest="tensor_shape_range",
        default="1,64",
        type=lambda x: str_to_list(x),
        help="Min,Max range of tensor shapes",
    )

    parser.add_argument(
        "--max-batch-size",
        dest="max_batch_size",
        default=1,
        type=int,
        help="Maximum batch size for NHWC tests",
    )

    parser.add_argument(
        "--max-conv-padding",
        dest="max_conv_padding",
        default=1,
        type=int,
        help="Maximum padding for Conv tests",
    )

    parser.add_argument(
        "--max-conv-dilation",
        dest="max_conv_dilation",
        default=2,
        type=int,
        help="Maximum dilation for Conv tests",
    )

    parser.add_argument(
        "--max-conv-stride",
        dest="max_conv_stride",
        default=2,
        type=int,
        help="Maximum stride for Conv tests",
    )

    parser.add_argument(
        "--max-pooling-padding",
        dest="max_pooling_padding",
        default=1,
        type=int,
        help="Maximum padding for pooling tests",
    )

    parser.add_argument(
        "--max-pooling-stride",
        dest="max_pooling_stride",
        default=2,
        type=int,
        help="Maximum stride for pooling tests",
    )

    parser.add_argument(
        "--max-pooling-kernel",
        dest="max_pooling_kernel",
        default=3,
        type=int,
        help="Maximum kernel for pooling tests",
    )

    parser.add_argument(
        "--num-rand-permutations",
        dest="num_rand_permutations",
        default=6,
        type=int,
        help="Number of random permutations for a given shape/rank for randomly-sampled parameter spaces",
    )

    parser.add_argument(
        "--max-resize-output-dim",
        dest="max_resize_output_dim",
        default=1000,
        type=int,
        help="Upper limit on width and height output dimensions for `resize` op. Default: 1000",
    )

    # Targetting a specific shape/rank/dtype
    parser.add_argument(
        "--target-shape",
        dest="target_shapes",
        action="append",
        default=[],
        type=lambda x: str_to_list(x),
        help="Create tests with a particular input tensor shape, e.g., 1,4,4,8 (may be repeated for tests that require multiple input shapes)",
    )

    parser.add_argument(
        "--target-rank",
        dest="target_ranks",
        action="append",
        default=None,
        type=lambda x: auto_int(x),
        help="Create tests with a particular input tensor rank",
    )

    parser.add_argument(
        "--target-dtype",
        dest="target_dtypes",
        action="append",
        default=None,
        type=lambda x: dtype_str_to_val(x),
        help="Create test with a particular DType (may be repeated)",
    )

    parser.add_argument(
        "--num-const-inputs-concat",
        dest="num_const_inputs_concat",
        default=0,
        choices=[0, 1, 2, 3],
        type=int,
        help="Allow constant input tensors for concat operator",
    )

    parser.add_argument(
        "--test-type",
        dest="test_type",
        choices=["positive", "negative", "both"],
        default="positive",
        type=str,
        help="type of tests produced, positive, negative, or both",
    )

    parser.add_argument(
        "--allow-pooling-and-conv-oversizes",
        dest="oversize",
        action="store_true",
        help="allow oversize padding, stride and kernel tests",
    )

    parser.add_argument(
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

    args = parser.parse_args(argv)

    return args


def main(argv=None):

    args = parseArgs(argv)

    ttg = TosaTestGen(args)

    if args.test_type == "both":
        testType = ["positive", "negative"]
    else:
        testType = [args.test_type]
    results = []
    for test_type in testType:
        testList = []
        for op in ttg.TOSA_OP_LIST:
            if re.match(args.filter + ".*", op):
                testList.extend(
                    ttg.genOpTestList(
                        op,
                        shapeFilter=args.target_shapes,
                        rankFilter=args.target_ranks,
                        dtypeFilter=args.target_dtypes,
                        testType=test_type,
                    )
                )

        print("{} matching {} tests".format(len(testList), test_type))

        testStrings = []
        for opName, testStr, dtype, error, shapeList, testArgs in testList:
            # Check for and skip duplicate tests
            if testStr in testStrings:
                print(f"Skipping duplicate test: {testStr}")
                continue
            else:
                testStrings.append(testStr)

            results.append(
                ttg.serializeTest(opName, testStr, dtype, error, shapeList, testArgs)
            )

    print(f"Done creating {len(results)} tests")


if __name__ == "__main__":
    exit(main())

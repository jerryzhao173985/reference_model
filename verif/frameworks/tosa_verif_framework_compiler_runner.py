#!/usr/bin/env python3
# Copyright (c) 2020-2024, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import math
import os
import queue
import re
import sys
import threading
import traceback
from datetime import datetime
from enum import IntEnum
from enum import unique
from pathlib import Path

import numpy as np
from checker.color_print import LogColors
from checker.color_print import print_color
from checker.color_print import set_print_in_color
from runner.run_command import run_sh_command
from xunit.xunit import xunit_results
from xunit.xunit import xunit_test


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--test",
        dest="test",
        default=[],
        type=Path,
        nargs="+",
        help="Test(s) to run",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        dest="recursive_tests",
        action="store_true",
        help="Recursively search for tests",
    )
    parser.add_argument(
        "--tf-base-dir",
        dest="tf_base_dir",
        type=str,
        required=True,
        help="Tensorflow/MLIR base directory",
    )
    parser.add_argument(
        "--tools-base-dir",
        dest="tools_base_dir",
        type=Path,
        required=True,
        help="Reference model base directory",
    )
    parser.add_argument(
        "-p",
        "--precise-mode",
        dest="precise_mode",
        action="store_true",
        help="run in precise mode (FP64)",
    )
    parser.add_argument(
        "-v", "--verbose", dest="verbose", action="count", help="Verbose run"
    )
    parser.add_argument(
        "-dref",
        "--debug-ref-model",
        dest="debug_ref_model",
        action="store_true",
        help="Enable TOSA Reference model debugging",
    )
    parser.add_argument(
        "--tolerance",
        dest="tolerance",
        default=1e-3,
        type=float,
        help="Comparison tolerance b value",
    )
    parser.add_argument(
        "--tosa_level",
        dest="tosa_level",
        default="EIGHTK",
        type=str,
        help="A TOSA level defines operator parameter ranges that an implementation shall support."
        "Config tosa_level for running the reference model only. Default is EIGHTK",
    )
    parser.add_argument(
        "--no-compiler",
        dest="no_compiler",
        action="store_true",
        help="Do not run TF MLIR/tfopt/TOSA compiler.  Just run TOSA Reference model",
    )
    parser.add_argument(
        "--no-ref-model",
        dest="no_ref",
        action="store_true",
        help="Do not run TOSA reference model, just run TF MLIR/tfopt/TOSA compiler.",
    )
    parser.add_argument(
        "--valgrind",
        dest="valgrind",
        action="store_true",
        help="Enable valgrind on TOSA Reference Model",
    )
    parser.add_argument(
        "-j", "--jobs", dest="jobs", type=int, default=1, help="Number of parallel jobs"
    )
    parser.add_argument(
        "--no-color",
        "--no-colour",
        dest="no_color",
        action="store_true",
        help="Disable color output",
    )
    parser.add_argument(
        "-f",
        "--framework",
        dest="framework",
        default=[],
        action="append",
        help="Frameworks to test (tf, tflite)",
    )
    parser.add_argument(
        "--override-exclusions",
        dest="override_exclusions",
        default=False,
        action="store_true",
        help="Ignore the framework exclusions listed in the test JSON",
    )
    parser.add_argument(
        "--xunit-file",
        dest="xunit_file",
        type=str,
        default="result.xml",
        help="XUnit result output file",
    )
    parser.add_argument(
        "--xunit-classname-prefix",
        dest="xunit_classname_prefix",
        default="TFUnitTests",
        help="Prefix for xunit classname",
    )
    parser.add_argument(
        "--hex-bool-hack",
        dest="hex_bool_hack",
        default=1,
        type=int,
        help=(
            "Hack around bug in MLIR hex parsing for boolean types"
            " by disabling hex encoding"
        ),
    )
    parser.add_argument(
        "--regression-mode",
        dest="regression_mode",
        default=False,
        action="store_true",
        help="Options to make the script more friendly for jenkins regressions",
    )
    parser.add_argument(
        "--quantize-tolerance",
        dest="quantize_tolerance",
        default=0,
        type=int,
        help=(
            "Tolerance when comparing TOSA reference model result"
            " to TensorFlow Lite reference"
        ),
    )
    parser.add_argument(
        "--test-dir",
        dest="test_dir",
        type=Path,
        help="Path to prepend to paths in test.json",
    )

    parser.add_argument(
        "-o", "--output", dest="output_file", help="Redirect script output to a file"
    )

    args = parser.parse_args()

    # No easy way to both do array append and override a default value
    if not args.framework:
        args.framework = ["tf", "tflite"]

    # Autodetect CPU count
    if args.jobs <= 0:
        args.jobs = os.cpu_count()

    return args


@unique
class TestResult(IntEnum):
    PASS = 0
    COMPILER_ERROR = 1
    REF_MODEL_ERROR = 2
    REF_MODEL_UNPREDICTABLE = 3
    REF_MODEL_RUNTIME_ERROR = 4
    MISMATCH = 5
    NOT_LOWERED = 6
    INVALID_MLIR = 7
    INTERNAL_ERROR = 8
    SKIPPED = 9


TestResultErrorStr = [
    "",
    "Compiler error",
    "Reference model error",
    "Reference model unpredictable",
    "Reference model runtime error",
    "Mismatch",
    "Not lowered",
    "Invalid MLIR",
    "Internal error",
    "",
]


def parse_compiler_output(compiler_stdout, compiler_stderr):
    # Look for "has not been lowered yet, skipped" strings in stdout
    expr = re.compile(".* has not been lowered yet, skipped.*")

    for line in compiler_stdout.splitlines():
        if expr.match(line):
            return TestResult.NOT_LOWERED

    return TestResult.PASS


def parse_reference_model_output(ref_model_stdout, ref_model_stderr):
    # Look for "has not been lowered yet, skipped" strings in stdout
    unpredictable_expr = re.compile(r".*UNPREDICTABLE.*")
    error_expr = re.compile(".* Graph result: ERROR.*")
    unknown_expr = re.compile(".* Unknown graph status code.*")

    for line in ref_model_stderr.splitlines():
        if unpredictable_expr.match(line):
            return TestResult.REF_MODEL_UNPREDICTABLE
        elif error_expr.match(line):
            return TestResult.REF_MODEL_ERROR
        elif unknown_expr.match(line):
            return TestResult.REF_MODEL_RUNTIME_ERROR

    return TestResult.PASS


# write a self-contained test descriptor in json format
def write_reference_runner_json(
    filename,
    tosa_filename,
    ifm_name,
    ifm_file,
    ofm_name,
    ofm_file,
    variable_name,
    variable_file,
    expected_failure=False,
):
    """Write a json test file so that it is fairly easy to pick up the test
    and generate commands for third party tool"""
    test_desc = dict()

    test_desc["tosa_file"] = tosa_filename
    test_desc["ifm_name"] = ifm_name
    test_desc["ifm_file"] = ifm_file
    test_desc["ofm_name"] = ofm_name
    test_desc["ofm_file"] = ofm_file
    test_desc["variable_name"] = variable_name
    test_desc["variable_file"] = variable_file
    test_desc["expected_failure"] = expected_failure

    with open(filename, "w") as f:
        json.dump(test_desc, f, indent="  ")


""" For dynamic shape model, apply 2 steps to perform compilation, shape inference,
    and serialization."""


def compile_dynamic_model(
    args,
    framework,
    test_path,
    test_name,
    pre_opt_filename,
    post_opt_filename,
    tosa_mlir_filename,
    compiler_cmd,
    flatbuffer_dir_fullpath,
    shape,
):
    try:
        # 1. Compile the dynamic shape model with unknown shapes and tosa shape ops.
        dyn_tosa_mlir_filename = str(test_path / f"output_{framework}.dyn.tosa.mlir")
        compile_dynamic_cmd = compiler_cmd.copy()
        compile_dynamic_cmd.extend(
            [
                "--verify-each",
                post_opt_filename,
                "-o",
                dyn_tosa_mlir_filename,
            ]
        )
        compiler_stdout, compiler_stderr = run_sh_command(
            compile_dynamic_cmd, args.verbose, True
        )

        compiler_rc_1 = parse_compiler_output(compiler_stdout, compiler_stderr)

        if compiler_rc_1 == TestResult.NOT_LOWERED:
            print_color(
                LogColors.RED,
                f"Results NOT_LOWERED {test_name}, framework {framework}",
            )
            return (TestResult.NOT_LOWERED, 0.0, "", test_name)

        def convert_shape_tuple_to_string(tup):
            string = ""
            for dim in tup:
                string = string + str(dim) + ","
            # skip the last `,` character.
            return string[0:-1]

        # 2. Resolve unknown shapes, and perform serialization.
        if not isinstance(shape, tuple):
            raise Exception("Only single input is supported currently")

        arg0_argument = '"arg0=' + convert_shape_tuple_to_string(shape) + '"'

        compile_and_shape_infer_cmd = compiler_cmd.copy()
        compile_and_shape_infer_cmd.extend(
            [
                f"--tosa-input-shape={arg0_argument}",
                "--tosa-infer-shapes",
                dyn_tosa_mlir_filename,
                "-o",
                tosa_mlir_filename,
                "--tosa-serialize",
                f"--tosa-flatbuffer-filename={flatbuffer_dir_fullpath / f'{test_name}.tosa'}",
            ]
        )

        # Convert list type to string type as double quote \" in list structure causes
        # single quote \' residue in the final command.
        compiler_stdout, compiler_stderr = run_sh_command(
            " ".join(map(str, compile_and_shape_infer_cmd)), args.verbose, True
        )

        compiler_rc_2 = parse_compiler_output(compiler_stdout, compiler_stderr)

        if compiler_rc_2 == TestResult.NOT_LOWERED:
            print_color(
                LogColors.RED,
                f"Results NOT_LOWERED {test_name}, framework {framework}",
            )
            return (TestResult.NOT_LOWERED, 0.0, "", test_name)

    except Exception as e:
        if "same scale constraint" in str(e):
            print_color(LogColors.RED, f"Results INVALID_MLIR {test_name}: {e}")
            return (TestResult.INVALID_MLIR, 0.0, e, test_name)
        else:
            print_color(LogColors.RED, f"Results COMPILER_ERROR {test_name}: {e}")
            return (TestResult.COMPILER_ERROR, 0.0, e, test_name)


def run_test(args, test_path, framework):
    msg = ""

    try:
        with open(test_path / "test.json", "r") as f:
            test_desc = json.load(f)
    except Exception:
        raise Exception(f"Could not load or parse test from {test_path / 'test.json'}")

    test_name = None
    if "name" in test_desc:
        test_name = test_desc["name"]
    else:
        test_name = test_path.name
    if not test_name:
        raise Exception(f"Could not parse test_name from {test_path}")

    print_color(LogColors.GREEN, f"## Running {framework} test {test_name}")

    try:
        if not args.override_exclusions:
            for excl in test_desc["framework_exclusions"]:
                if excl == framework:
                    print_color(LogColors.GREEN, "Results SKIPPED")
                    return (TestResult.SKIPPED, 0.0, "", test_name)
    except KeyError:
        pass

    tf_tools_dir = Path(
        f"{args.tf_base_dir}/bazel-bin/tensorflow/compiler/mlir"
    ).resolve()

    pre_opt_filename = str(test_path / f"test_{framework}.preopt.mlir")
    post_opt_filename = str(test_path / f"test_{framework}.postopt.mlir")
    if args.test_dir:
        test_path_prepend = args.test_dir
    else:
        test_path_prepend = test_path

    # 1. Framework to MLIR translator command
    if framework == "tf":
        if test_desc["tf_model_filename"].endswith(".mlir"):
            pre_opt_filename = test_desc["tf_model_filename"]
            translate_mlir_cmd = []
        else:
            translate_mlir_cmd = [
                str(tf_tools_dir / "tf-mlir-translate"),
                "--graphdef-to-mlir",
                "--tf-enable-shape-inference-on-import",
                f"--tf-output-arrays={test_desc['tf_result_name']}",
                str(test_path_prepend / test_desc["tf_model_filename"]),
                "-o",
                pre_opt_filename,
            ]
    elif framework == "tflite":
        if test_desc["tflite_model_filename"].endswith(".mlir"):
            pre_opt_filename = test_desc["tflite_model_filename"]
            translate_mlir_cmd = []
        else:
            translate_mlir_cmd = [
                str(tf_tools_dir / "lite" / "flatbuffer_translate"),
                "--tflite-flatbuffer-to-mlir",
                str(test_path_prepend / test_desc["tflite_model_filename"]),
                f"--output-arrays={test_desc['tflite_result_name']}",
                "-o",
                pre_opt_filename,
            ]
    else:
        raise Exception(f"Unknown framwork: {framework}")

    # Any additional inputs to the translator?
    input_tensor_prefix = "TosaInput_"
    flatbuffer_dir = f"flatbuffer-{framework}"
    mlir_opts = []

    # Temporary hack: MLIR's new hex encoding of large tensors does not work for
    # boolean types
    # for TF hash 8e8041d594a888eb67eafa5cc62627d7e9ca8082
    if str(test_path).endswith("_bool") and args.hex_bool_hack:
        mlir_opts.append("--mlir-print-elementsattrs-with-hex-if-larger=-1")

    try:
        # specify input tensors if test is generated from .pb
        if framework == "tf":
            # Convert the shape to a mlir-friendly string
            shapes = []
            for curr_shape in test_desc["ifm_shape"]:
                shape_str = ""
                for dim in curr_shape:
                    shape_str = shape_str + str(dim) + ","
                shapes.append(shape_str)

            translate_mlir_cmd.extend(
                ["--tf-input-arrays", ",".join(test_desc["ifm_name"])]
            )
            translate_mlir_cmd.extend(["--tf-input-shapes", ":".join(shapes)])

        # Write the hard-coded placeholder input (reshaped as necesary) to
        # the file that compiler specified.
        reference_runner_ifm_name = []
        for i in range(len(test_desc["ifm_file"])):
            ifm_tensor_name = f"{input_tensor_prefix}{i}"

            assert test_desc["ifm_file"][i].endswith(".npy")
            ifm_np = np.load(test_path / test_desc["ifm_file"][i])

            # We sometimes encounter input shape/expected input shape mismatches
            # due to a missing batch dimension on the input (e.g. a single 3D image).
            #
            # Make sure input numpy and input shape from descriptor match,
            # expand_dims on the outer dimensions until the rank matches,
            # then do the shape comparison.
            while len(list(ifm_np.shape)) < len(test_desc["ifm_shape"][i]):
                ifm_np = np.expand_dims(ifm_np, axis=0)

            # After legalization, complex tensors are expected to be represented
            # as a single floating point tensor of shape [?, ..., ?, 2].
            expected_shape = test_desc["ifm_shape"][i]
            if str(test_path).endswith("c64"):
                expected_shape.append(2)

            assert list(ifm_np.shape) == expected_shape

            reference_runner_ifm_name.append(ifm_tensor_name)

    except KeyError:
        # No additional inputs.  Ignore.
        pass

    tf_opt_cmd = [
        str(tf_tools_dir / "tf-opt"),
        "--tf-executor-to-functional-conversion",
        "--verify-each",
        pre_opt_filename,
        "-o",
        post_opt_filename,
    ]

    translate_mlir_cmd.extend(mlir_opts)
    tf_opt_cmd.extend(mlir_opts)

    compiler_cmd = [str(tf_tools_dir / "tf-opt")]

    if framework == "tf":
        compiler_cmd.append("--tf-to-tosa-pipeline")
    elif framework == "tflite":
        compiler_cmd.append("--tfl-to-tosa-pipeline")
        compiler_cmd.append("--tosa-strip-quant-types")

    tosa_mlir_filename = str(test_path / f"output_{framework}.tosa.mlir")

    flatbuffer_dir_fullpath = test_path / flatbuffer_dir

    flatbuffer_dir_fullpath.mkdir(exist_ok=True)

    compile_and_serialize_cmd = compiler_cmd.copy()
    compile_and_serialize_cmd.extend(
        [
            "--verify-each",
            post_opt_filename,
            "-o",
            tosa_mlir_filename,
            "--tosa-serialize",
            f"--tosa-flatbuffer-filename={flatbuffer_dir_fullpath / f'{test_name}.tosa'}",
        ]
    )

    if not args.no_compiler:
        try:
            if translate_mlir_cmd:
                run_sh_command(translate_mlir_cmd, args.verbose, True)
            if tf_opt_cmd:
                run_sh_command(tf_opt_cmd, args.verbose, True)
        except Exception as e:
            print_color(LogColors.RED, f"Results INVALID_MLIR {test_name}: {e}")
            return (TestResult.INVALID_MLIR, 0.0, e, test_name)

        if "ifm_dynamic" in test_desc and test_desc["ifm_dynamic"] == 1:
            compile_dynamic_model(
                args,
                framework,
                test_path,
                test_name,
                pre_opt_filename,
                post_opt_filename,
                tosa_mlir_filename,
                compiler_cmd,
                flatbuffer_dir_fullpath,
                ifm_np.shape,
            )
        else:
            try:
                compiler_stdout, compiler_stderr = run_sh_command(
                    compile_and_serialize_cmd, args.verbose, True
                )
                compiler_rc = parse_compiler_output(compiler_stdout, compiler_stderr)
                if compiler_rc == TestResult.NOT_LOWERED:
                    print_color(
                        LogColors.RED,
                        f"Results NOT_LOWERED {test_name}, framework {framework}",
                    )
                    return (TestResult.NOT_LOWERED, 0.0, "", test_name)

                pass

            except Exception as e:
                if "same scale constraint" in str(e):
                    print_color(LogColors.RED, f"Results INVALID_MLIR {test_name}: {e}")
                    return (TestResult.INVALID_MLIR, 0.0, e, test_name)
                else:
                    print_color(
                        LogColors.RED, f"Results COMPILER_ERROR {test_name}: {e}"
                    )
                    return (TestResult.COMPILER_ERROR, 0.0, e, test_name)

    if framework == "tf":
        try:
            tf_result = np.load(test_path / test_desc["tf_result_npy_filename"])
        except KeyError:
            assert 0, "fail to load tf result numpy"
    elif framework == "tflite":
        try:
            tf_result = np.load(test_path / test_desc["tflite_result_npy_filename"])
        except KeyError:
            assert 0, "fail to load tflite result numpy"

    # TOSA has no notion of complex datatypes, it represents complex values using two
    # fp32 output tensors representing real and imaginary values. When legalizing
    # complex operations from frameworks, these two output tensors are combined into
    # a single tensor of shape [?, ..., ?, 2] whereby each inner pair of values
    # represents the real and imaginary parts of a complex value. This is completed
    # by inserting reshape and concatenate TOSA operations during the legalization to
    # maintain a one-to-one correspondance with framework outputs, thus simplifying
    # legalization. Here tf_result should also match this format before being
    # compared to the ref model output.
    if tf_result.dtype == np.complex64:
        ifm_shape = tf_result.shape + (2,)
        tf_result = tf_result.view(np.float32)
        tf_result = tf_result.reshape(ifm_shape)

    # Generate test descriptor per flatbuffer generation
    # Input .npy will be shared across different frameworks
    # Output .npy will be generated in its corresponding flatbuffer
    reference_runner_ifm_file = [
        str(Path("..") / ifm_file) for ifm_file in test_desc["ifm_file"]
    ]

    # Check if there's any operator in output graph.
    empty_graph = True
    with open(tosa_mlir_filename, "r") as f:
        for line in f:
            # TOSA assembly instructions all start with `tosa.`
            if re.search(r"tosa\.", line):
                empty_graph = False

                break

    # Fast-forward input tensor to output tensor if TOSA graph is empty.
    if empty_graph:
        reference_runner_ofm_name = reference_runner_ifm_name
    else:
        reference_runner_ofm_name = ["TosaOutput_0"]

    if "num_variables" in test_desc:
        num_variable = test_desc["num_variables"]
    else:
        num_variable = 0
    reference_runner_variable_name = []
    reference_runner_variable_file = []

    for i in range(num_variable):
        variable_name_str = "Variable_" + str(i)
        variable_file_str = "variable_output_" + str(i) + ".npy"
        reference_runner_variable_name.append(variable_name_str)
        reference_runner_variable_file.append(variable_file_str)

    write_reference_runner_json(
        filename=str(test_path / flatbuffer_dir / "desc.json"),
        tosa_filename=f"{test_name}.tosa",
        ifm_name=reference_runner_ifm_name,
        ifm_file=reference_runner_ifm_file,
        ofm_name=reference_runner_ofm_name,
        ofm_file=["ref_model_output_0.npy"],
        variable_name=reference_runner_variable_name,
        variable_file=reference_runner_variable_file,
    )

    ref_model_cmd = [
        str(args.tools_base_dir / "build" / "reference_model" / "tosa_reference_model"),
        f"--test_desc={test_path / flatbuffer_dir / 'desc.json'}",
    ]

    if args.debug_ref_model:
        ref_model_cmd.extend(["-D ALL", "-l high"])

    if args.precise_mode:
        ref_model_cmd.extend(["--precise_mode=1"])

    if args.valgrind:
        ref_model_cmd = [
            "valgrind",
            "--show-leak-kinds=all",
            "--log-fd=1",
            "-q",
        ] + ref_model_cmd

    ref_model_cmd = ref_model_cmd + [f"--tosa_level={args.tosa_level}"]

    # Clean out any ref_model result first
    for f in (test_path / flatbuffer_dir).glob("ref_model_*.npy"):
        f.unlink()

    if args.no_ref:
        return (TestResult.PASS, 0.0, msg)

    try:
        ref_model_stdout, ref_model_stderr = run_sh_command(
            ref_model_cmd, args.verbose, True
        )
        ref_model_rc = parse_reference_model_output(ref_model_stdout, ref_model_stderr)
        if ref_model_rc != TestResult.PASS:
            return (ref_model_rc, 0.0, "")
    except Exception as e:
        ref_model_rc = parse_reference_model_output("", str(e))
        if ref_model_rc != TestResult.PASS:
            print_color(
                LogColors.RED,
                f"Results {TestResultErrorStr[ref_model_rc]} {test_name}: {e}",
            )
            return (ref_model_rc, 0.0, "")
        print_color(LogColors.RED, f"Results REF_MODEL_RUNTIME_ERROR {test_name}: {e}")
        return (TestResult.REF_MODEL_RUNTIME_ERROR, 0.0, e, test_name)

    if args.precise_mode == 1 and (
        tf_result.dtype == np.float16 or tf_result.dtype == np.float32
    ):
        tf_result = tf_result.astype(np.float64)
    elif tf_result.dtype == np.float16:
        tf_result = tf_result.astype(np.float32)
    elif tf_result.dtype == np.int8:
        tf_result = tf_result.astype(np.int8)
    elif tf_result.dtype == np.uint8:
        tf_result = tf_result.astype(np.uint8)
    elif tf_result.dtype == np.int16:
        tf_result = tf_result.astype(np.int16)
    elif tf_result.dtype == np.uint16:
        tf_result = tf_result.astype(np.uint16)
    elif tf_result.dtype == np.int64:
        tf_result = tf_result.astype(np.int32)

    # For now, search for the first output from ref_model
    ref_model_result_files = list((test_path / flatbuffer_dir).glob("ref_model_*.npy"))
    ref_model_result = np.load(ref_model_result_files[0])

    if np.issubdtype(tf_result.dtype, np.unsignedinteger) and (
        tf_result.dtype != ref_model_result.dtype
    ):
        ref_model_result = ref_model_result.astype(tf_result.dtype)

    assert (
        tf_result.dtype == ref_model_result.dtype
    ), f"Numpy type mismatch {tf_result.dtype} != {ref_model_result.dtype} when comparing result"

    # Size comparison
    # Size = 1 tensors can be equivalently represented as having rank 0 or rank
    # >= 0, allow that special case
    tf_result = np.squeeze(tf_result)
    ref_model_result = np.squeeze(ref_model_result)

    if np.shape(tf_result) != np.shape(ref_model_result):
        print_color(LogColors.RED, f"Results MISCOMPARE {test_name}")
        msg = f"Shapes mismatch: Reference {np.shape(tf_result)} vs {np.shape(ref_model_result)}"
        print(msg)
        return (TestResult.MISMATCH, 0.0, msg, test_name)

    # for quantized test, allow +-(args.quantize_tolerance) error
    if ref_model_result.dtype == np.int32:
        assert tf_result.dtype == np.int32

        if np.all(np.absolute(ref_model_result - tf_result) <= args.quantize_tolerance):
            print_color(LogColors.GREEN, f"Results PASS {test_name}")
        else:
            print_color(LogColors.RED, f"Results MISCOMPARE {test_name}")

            tolerance = args.quantize_tolerance + 1
            while not np.all(
                np.absolute(ref_model_result - tf_result) <= args.quantize_tolerance
            ):
                tolerance = tolerance + 1
                if tolerance >= 10:
                    break

            msg = f"Result is within {tolerance} {test_path}"
            print(msg)

            np.set_printoptions(threshold=128)
            print(f"tf_result: {tf_result.shape}\n")
            print(tf_result)
            print(f"ref_model_result: {ref_model_result.shape}\n")
            print(ref_model_result)
            # print(tf_result - ref_model_result)
            return (TestResult.MISMATCH, tolerance, msg, test_name)
    else:
        if np.allclose(
            ref_model_result, tf_result, atol=args.tolerance, equal_nan=True
        ):
            print_color(LogColors.GREEN, f"Results PASS {test_name}")
        else:
            print_color(LogColors.RED, f"Results MISCOMPARE {test_name}")

            # Many of these tests would match with a reasonable looser tolerence.
            # Determine what would have worked.
            tolerance = args.tolerance * 10.0
            while not np.allclose(
                ref_model_result, tf_result, atol=tolerance, equal_nan=True
            ):
                tolerance = tolerance * 10.0
                if tolerance > 1.0e10:
                    tolerance = math.inf
                    break

            msg = f"Result is within {tolerance:.0e} {test_name}"
            print(msg)

            np.set_printoptions(precision=4, threshold=128)
            print(f"tf_result: {tf_result.shape}\n")
            print(tf_result)
            print(f"ref_model_result: {ref_model_result.shape}\n")
            print(ref_model_result)
            # print(tf_result - ref_model_result)
            return (TestResult.MISMATCH, tolerance, msg, test_name)

    return (TestResult.PASS, args.tolerance, msg, test_name)


def worker_thread(task_queue, args, result_queue):
    while True:
        try:
            (test, framework) = task_queue.get(block=False)
        except queue.Empty:
            break

        if test is None:
            break

        msg = ""
        start_time = datetime.now()
        try:
            (rc, tolerance, msg, test_name) = run_test(args, test, framework)
        except Exception as e:
            print(f"Internal regression error: {e}")
            print(
                "".join(
                    traceback.format_exception(
                        etype=type(e), value=e, tb=e.__traceback__
                    )
                )
            )
            rc = TestResult.INTERNAL_ERROR
            tolerance = 0.0

        end_time = datetime.now()

        result_queue.put(
            (test, framework, rc, tolerance, msg, end_time - start_time, test_name)
        )
        task_queue.task_done()

    return True


def getTestsInDir(directory):
    # Recursively find any tests in this directory
    if (directory / "test.json").is_file():
        return [directory]
    elif directory.is_dir():
        test_list = []
        for d in directory.glob("*"):
            test_list.extend(getTestsInDir(d))
        return test_list
    else:
        return []


def main():
    args = parse_args()

    set_print_in_color(not args.no_color)

    if args.output_file:
        set_print_in_color(False)
        sys.stdout = open(args.output_file, "w")

    # Disable TF info messages
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    task_queue = queue.Queue()
    result_queue = queue.Queue()

    threads = []

    # Result counters for each of the TestResult return codes
    results = [0] * len(TestResult)

    for tdir in args.test:
        if args.recursive_tests:
            tdirList = getTestsInDir(tdir)
        else:
            tdirList = [tdir]

        for t in tdirList:
            for f in args.framework:
                task_queue.put((t, f))

    for i in range(args.jobs):
        t = threading.Thread(
            target=worker_thread, args=(task_queue, args, result_queue)
        )
        t.setDaemon(True)
        t.start()
        threads.append(t)

    # Run until queue is empty
    task_queue.join()

    print_color(LogColors.BOLD_WHITE, "Result summary")

    result_list = []
    while True:
        try:
            test, framework, rc, tol, msg, time_delta, test_name = result_queue.get(
                block=False
            )
        except queue.Empty:
            break

        result_list.append((test, framework, rc, tol, msg, time_delta, test_name))
        results[rc] = results[rc] + 1

    xunit_result = xunit_results()
    xunit_suite = xunit_result.create_suite(args.xunit_classname_prefix)

    # Sort by test name
    for test, framework, rc, tol, err_msg, time_delta, test_name in sorted(
        result_list, key=lambda tup: tup[0]
    ):
        class_name = f"{args.xunit_classname_prefix}.{framework}"

        xt = xunit_test(test_name, class_name)

        msg = TestResultErrorStr[rc]

        xt.time = str(
            float(time_delta.seconds) + (float(time_delta.microseconds) * 1e-6)
        )

        if len(msg) > 0:
            print(f"{msg} on {framework} {test}")

        # Add any more verbose messaging for the xml log
        if err_msg:
            msg = f"{msg} {err_msg}"

        if rc == TestResult.PASS:
            pass
        elif rc == TestResult.SKIPPED:
            xt.skipped()
        else:
            xt.failed(msg)

        xunit_suite.tests.append(xt)

        result_queue.task_done()

    xunit_result.write_results(args.xunit_file)

    print("Totals: ", end="")
    for result in TestResult:
        print(f"{results[result]} {result.name.lower()}, ", end="")
    print()

    if not args.regression_mode and (
        results[TestResult.COMPILER_ERROR] > 0
        or results[TestResult.REF_MODEL_ERROR] > 0
        or results[TestResult.MISMATCH] > 0
    ):
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
